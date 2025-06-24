import asyncio
import logging
import time
from typing import Dict, Optional, Any, NamedTuple, DefaultDict, Callable, Awaitable
from collections import defaultdict
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
        self.samples_per_chunk = int(self.sample_rate * self.chunk_duration)


class NIDAQTaskGroup:
    def __init__(self, sample_rate: int, modules: Dict[str, NIDAQModule]):
        self.logger = logging.getLogger(f"pdx.nidaqgroup.{sample_rate}")
        self.sample_rate = sample_rate
        self.modules = modules
        self.task: Optional[nidaqmx.Task] = None
        self.reader: Optional[AnalogMultiChannelReader] = None
        self.buffer: Optional[np.ndarray] = None
        self.total_channels = sum(len(m.channels) for m in modules.values())
        self.samples_per_chunk = list(modules.values())[0].samples_per_chunk
        self.chunk_duration = list(modules.values())[0].chunk_duration
        self.module_channel_indices: Dict[str, slice] = {}

    async def setup_task(self) -> bool:
        try:
            task_name = f"TaskGroup_{self.sample_rate}Hz"
            self.logger.debug(f"Creating task: {task_name}")
            self.task = nidaqmx.Task(task_name)

            # Step 1: Add all channels to the task first
            all_channels_config = []
            current_channel_index = 0
            for slot_id, module in self.modules.items():
                num_module_channels = len(module.channels)
                self.module_channel_indices[slot_id] = slice(
                    current_channel_index, current_channel_index + num_module_channels
                )
                current_channel_index += num_module_channels

                for channel_id, config in module.channels.items():
                    physical_channel = f"{module.device}/{channel_id}"
                    self.logger.debug(f"Adding channel: {physical_channel}")
                    self.task.ai_channels.add_ai_voltage_chan(physical_channel)
                    all_channels_config.append(config)

            # Step 2: Configure timing for the entire task
            buffer_size = self.samples_per_chunk * 20
            self.logger.debug(f"Configuring timing: {self.sample_rate} Hz")
            master_clock_source = (
                f"/{list(self.modules.values())[0].device}/ai/SampleClock"
            )
            self.logger.info(f"Using {master_clock_source} as the master sample clock.")
            self.task.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                source=master_clock_source,
                active_edge=Edge.RISING,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=buffer_size,
            )

            # Step 3: Configure individual channel properties
            for ai_channel, config in zip(self.task.ai_channels, all_channels_config):
                coupling_type = (
                    Coupling.AC if config.coupling.upper() == "AC" else Coupling.DC
                )
                if hasattr(ai_channel, "ai_coupling"):
                    ai_channel.ai_coupling = coupling_type
                    self.logger.debug(
                        f"Set coupling for {ai_channel.name}: {coupling_type}"
                    )

                if coupling_type == Coupling.AC and hasattr(ai_channel, "ai_excit_src"):
                    if config.iepe:
                        ai_channel.ai_excit_src = ExcitationSource.INTERNAL
                        if hasattr(ai_channel, "ai_excit_val"):
                            ai_channel.ai_excit_val = config.iepe_current
                        self.logger.debug(f"Enabled IEPE for {ai_channel.name}")
                    else:
                        ai_channel.ai_excit_src = ExcitationSource.NONE
                        self.logger.debug(f"Disabled IEPE for {ai_channel.name}")

            self.logger.debug("Configuring input buffer")
            self.task.in_stream.input_buf_size = buffer_size * 2
            self.task.in_stream.auto_start = True
            self.task.in_stream.relative_to = ReadRelativeTo.CURRENT_READ_POSITION
            self.task.in_stream.offset = 0

            self.buffer = np.zeros(
                (self.total_channels, self.samples_per_chunk), dtype=np.float64
            )
            self.reader = AnalogMultiChannelReader(self.task.in_stream)

            self.logger.info(
                f"Successfully configured task group for {self.sample_rate} Hz with {self.total_channels} channels"
            )
            return True

        except errors.DaqError as e:
            self.logger.error(
                f"DAQmx Error setting up task group for {self.sample_rate} Hz: {e}"
            )
            if self.task:
                self.task.close()
            return False
        except Exception as e:
            self.logger.error(
                f"Unexpected error setting up task group for {self.sample_rate} Hz: {e}"
            )
            if self.task:
                self.task.close()
            return False

    def _read_chunk_blocking(self):
        if not self.task or not self.reader or self.buffer is None:
            return
        self.reader.read_many_sample(
            self.buffer,
            number_of_samples_per_channel=self.samples_per_chunk,
            timeout=self.chunk_duration * 5,
        )

    async def read_chunk(self) -> Optional[Dict[str, np.ndarray]]:
        if not self.task or not self.reader or self.buffer is None:
            return None
        try:
            await asyncio.to_thread(self._read_chunk_blocking)
            demultiplexed_data = {}
            for slot_id, channel_slice in self.module_channel_indices.items():
                demultiplexed_data[slot_id] = self.buffer[channel_slice, :].copy()
            return demultiplexed_data
        except Exception as e:
            self.logger.error(
                f"Error reading from task group {self.sample_rate} Hz: {e}"
            )
            return None


class AsyncNIDAQClient:
    def __init__(self, chassis: str, modules_config: Dict[str, Any]):
        self.logger = logging.getLogger("pdx-asyncua.nidaqclient")
        self.chassis = chassis
        self.modules: Dict[str, NIDAQModule] = {}
        self.task_groups: Dict[int, NIDAQTaskGroup] = {}
        self._running = False
        self.acquisition_tasks = []

        # Create module objects from config
        grouped_modules: DefaultDict[int, Dict[str, NIDAQModule]] = defaultdict(dict)
        for slot_id, config in modules_config.items():
            channels = {
                channel_id: ChannelConfig(
                    name=channel_info["name"],
                    coupling=channel_info.get("coupling", "DC"),
                    iepe=channel_info.get("iepe", False),
                )
                for channel_id, channel_info in config["channels"].items()
            }
            chunk_duration = 0.1 if config["sample_rate"] > 10000 else 0.5
            module = NIDAQModule(
                device=config["device"],
                channels=channels,
                sample_rate=config["sample_rate"],
                chunk_duration=chunk_duration,
                module_type=config["name"],
            )
            self.modules[slot_id] = module
            grouped_modules[module.sample_rate][slot_id] = module

        for sample_rate, modules in grouped_modules.items():
            self.task_groups[sample_rate] = NIDAQTaskGroup(sample_rate, modules)

    async def setup(self) -> bool:
        """Setup all NI-DAQ task groups"""
        setup_tasks = [group.setup_task() for group in self.task_groups.values()]
        results = await asyncio.gather(*setup_tasks)
        if not all(results):
            self.logger.error("Failed to setup one or more NI-DAQ task groups.")
            await self.cleanup()
            return False
        self.logger.info("All NI-DAQ task groups configured successfully.")
        return True

    async def start_acquisition(self) -> bool:
        """Start acquisition on all task groups"""
        try:
            for group in self.task_groups.values():
                if group.task:
                    group.task.start()
                    self.logger.info(
                        f"Started acquisition on task group {group.sample_rate} Hz"
                    )
            return True
        except Exception as e:
            self.logger.error(f"Error starting acquisition: {e}")
            return False

    async def stop_acquisition(self):
        """Stop acquisition and cleanup"""
        self._running = False
        for task in self.acquisition_tasks:
            if not task.done():
                task.cancel()
        if self.acquisition_tasks:
            await asyncio.gather(*self.acquisition_tasks, return_exceptions=True)

        for group in self.task_groups.values():
            if group.task:
                try:
                    group.task.stop()
                    group.task.close()
                    self.logger.info(
                        f"Stopped acquisition on task group {group.sample_rate} Hz"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Error stopping task group {group.sample_rate} Hz: {e}"
                    )
        self.task_groups.clear()

    async def _acquisition_loop(
        self,
        task_group: NIDAQTaskGroup,
        data_callback: Callable[[Dict[str, np.ndarray]], Awaitable[None]],
    ):
        """Acquisition loop for a single task group."""
        self.logger.info(
            f"Starting acquisition loop for sample rate {task_group.sample_rate} Hz"
        )
        while self._running:
            try:
                data = await task_group.read_chunk()
                if data:
                    await data_callback(data)
            except asyncio.CancelledError:
                self.logger.info(
                    f"Acquisition loop for {task_group.sample_rate} Hz cancelled."
                )
                break
            except Exception as e:
                self.logger.error(
                    f"Error in acquisition loop for {task_group.sample_rate} Hz: {e}"
                )
                await asyncio.sleep(0.1)

    async def run_acquisition(
        self, data_callback: Callable[[Dict[str, np.ndarray]], Awaitable[None]]
    ):
        """Run acquisition loops for all task groups."""
        self._running = True
        self.acquisition_tasks = []
        for task_group in self.task_groups.values():
            loop_task = asyncio.create_task(
                self._acquisition_loop(task_group, data_callback)
            )
            self.acquisition_tasks.append(loop_task)
        if self.acquisition_tasks:
            await asyncio.gather(*self.acquisition_tasks)

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
