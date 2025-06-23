import asyncio
import logging
import time
from typing import Dict, Optional, Any, NamedTuple
import numpy as np
import nidaqmx
from nidaqmx import errors
from nidaqmx.constants import (
    Edge,
    AcquisitionType,
    Coupling,
    ExcitationSource,
    ReadRelativeTo,
)
from nidaqmx.stream_readers import AnalogMultiChannelReader


class ChannelConfig(NamedTuple):
    """Channel configuration container"""

    name: str
    coupling: str
    iepe: bool
    iepe_current: float = 2.0e-3


class NIDAQModule:
    def __init__(
        self,
        device: str,
        channels: Dict[str, ChannelConfig],
        sample_rate: int,
        chunk_duration: float = 0.1,
        module_type: str = "",
    ):
        self.logger = logging.getLogger("pdx.asyncnidaqmodule")
        self.device = device
        self.channels = channels
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.module_type = module_type
        self.task: Optional[nidaqmx.Task] = None
        self.reader: Optional[AnalogMultiChannelReader] = None

        # Increase buffer sizes
        self.samples_per_chunk = int(self.sample_rate * self.chunk_duration)
        self.buffer_size = self.samples_per_chunk * 20
        self.buffer: Optional[np.ndarray] = None
        self.start_time: Optional[float] = None
        self.total_samples = 0

    async def setup_task(self) -> bool:
        """Setup the DAQ task with proper channel configuration"""
        try:
            task_name = f"{self.device}_AI_Task"
            self.logger.debug(f"Creating task: {task_name}")
            self.task = nidaqmx.Task(task_name)

            for channel_id in self.channels:
                physical_channel = f"{self.device}/{channel_id}"
                self.logger.debug(f"Adding channel: {physical_channel}")
                self.task.ai_channels.add_ai_voltage_chan(physical_channel)

            self.logger.debug(f"Configuring timing: {self.sample_rate} Hz")
            self.task.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                source="",
                active_edge=Edge.RISING,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.buffer_size,
            )

            for channel, (channel_id, config) in zip(
                self.task.ai_channels, self.channels.items()
            ):
                coupling_type = Coupling.DC

                if hasattr(channel, "ai_coupling"):
                    coupling_type = (
                        Coupling.AC if config.coupling.upper() == "AC" else Coupling.DC
                    )
                    channel.ai_coupling = coupling_type
                    self.logger.debug(f"Set coupling for {channel_id}: {coupling_type}")

                if coupling_type == Coupling.AC:
                    if hasattr(channel, "ai_excit_src"):
                        if config.iepe:
                            channel.ai_excit_src = ExcitationSource.INTERNAL
                            if hasattr(channel, "ai_excit_val"):
                                channel.ai_excit_val = config.iepe_current
                            self.logger.debug(f"Enabled IEPE for {channel_id}")
                        else:
                            channel.ai_excit_src = ExcitationSource.NONE
                            self.logger.debug(f"Disabled IEPE for {channel_id}")

            # Configure larger input buffer
            self.logger.debug("Configuring input buffer")
            self.task.in_stream.input_buf_size = (
                self.buffer_size * 1  # 8
            )  # Increased multiplier
            self.task.in_stream.auto_start = True  # Changed to True
            self.task.in_stream.relative_to = ReadRelativeTo.CURRENT_READ_POSITION
            self.task.in_stream.offset = 0

            # Initialize the data buffer and reader
            self.buffer = np.zeros(
                (len(self.channels), self.samples_per_chunk), dtype=np.float64
            )
            self.reader = AnalogMultiChannelReader(self.task.in_stream)

            self.logger.info(
                f"Successfully configured task for {self.device} with "
                f"{len(self.channels)} channels at {self.sample_rate} Hz"
            )
            return True

        except errors.DaqError as e:
            self.logger.error(
                f"DAQmx Error setting up task for {self.device}: {str(e)}"
            )
            if self.task:
                self.task.close()
                self.task = None
            return False
        except Exception as e:
            self.logger.error(
                f"Unexpected error setting up task for {self.device}: {str(e)}"
            )
            if self.task:
                self.task.close()
                self.task = None
            return False

    def _read_chunk_blocking(self) -> None:
        """Blocking call to read data from the DAQ"""
        if not self.task or not self.reader or self.buffer is None:
            return

        self.reader.read_many_sample(
            self.buffer,
            number_of_samples_per_channel=self.samples_per_chunk,
            timeout=self.chunk_duration * 2,
        )

    async def read_chunk(self) -> Optional[np.ndarray]:
        """Read a chunk of data from the DAQ"""
        if not self.task or not self.reader or self.buffer is None:
            return None

        try:
            if self.start_time is None:
                self.start_time = time.time()

            await asyncio.to_thread(self._read_chunk_blocking)
            self.total_samples += self.samples_per_chunk
            return self.buffer.copy()
        except Exception as e:
            self.logger.error(f"Error reading from module {self.module_type}: {str(e)}")
            return None


class AsyncNIDAQClient:
    def __init__(self, chassis: str, modules_config: Dict[str, Any]):
        self.logger = logging.getLogger("pdx-asyncua.nidaqclient")
        self.chassis = chassis
        self.modules = {}

        # Create module objects from config
        for slot_id, config in modules_config.items():
            # Convert channel config to ChannelConfig objects
            channels = {}
            for channel_id, channel_info in config["channels"].items():
                if isinstance(channel_info, dict):
                    channels[channel_id] = ChannelConfig(
                        name=channel_info["name"],
                        coupling=channel_info.get("coupling", "DC"),
                        iepe=channel_info.get("iepe", False),
                        # iepe_current=channel_info.get("iepe_current", 2.0),
                    )
                else:
                    # Handle legacy format where channel_info is just the name
                    channels[channel_id] = ChannelConfig(
                        name=channel_info,
                        coupling="DC",
                        iepe=False,
                        # iepe_current=2.0,
                    )

            # Adjust chunk duration based on sample rate
            chunk_duration = 0.1 if config["sample_rate"] > 10000 else 0.5

            module = NIDAQModule(
                device=config["device"],
                channels=channels,
                sample_rate=config["sample_rate"],
                chunk_duration=chunk_duration,
                module_type=config["name"],
            )
            self.modules[slot_id] = module

    async def setup(self) -> bool:
        """Setup all NI-DAQ modules"""
        try:
            for slot_id, module in self.modules.items():
                if not await module.setup_task():
                    self.logger.error(f"Failed to setup module {slot_id}")
                    return False
                self.logger.info(f"Successfully configured module {slot_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error setting up NI-DAQ: {e}")
            await self.cleanup()
            return False

    async def start_acquisition(self) -> bool:
        """Start acquisition on all modules"""
        try:
            for slot_id, module in self.modules.items():
                if module.task:
                    module.task.start()
                    self.logger.info(f"Started acquisition on module {slot_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error starting acquisition: {e}")
            return False

    async def stop_acquisition(self):
        """Stop acquisition and cleanup"""
        for slot_id, module in self.modules.items():
            if module.task:
                try:
                    module.task.stop()
                    module.task.close()
                    self.logger.info(f"Stopped acquisition on module {slot_id}")
                except Exception as e:
                    self.logger.error(f"Error stopping module {slot_id}: {e}")
        self.modules.clear()

    async def read_chunk_from_all_modules(self) -> Dict[str, np.ndarray]:
        """Read a chunk of data from all modules"""
        data = {}
        try:
            read_tasks = [module.read_chunk() for module in self.modules.values()]
            results = await asyncio.gather(*read_tasks)
            for (slot_id, _), result in zip(self.modules.items(), results):
                if result is not None:
                    data[slot_id] = result
            return data
        except Exception as e:
            self.logger.error(f"Error reading data: {e}")
            return {}

    async def cleanup(self):
        """Cleanup resources"""
        await self.stop_acquisition()

    async def __aenter__(self):
        """Async context manager entry"""
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
