from alive_progress import alive_bar
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, cast
import h5py
from h5py import Group, Dataset
import numpy as np

from .asyncuaclient import AsyncOPCUAClient
from .asyncnidaqclient import AsyncNIDAQClient
from .utils import get_utcnow


class DataCollector:
    def __init__(self, config: dict):
        """Initialize DataCollector with configuration"""
        self.logger = logging.getLogger("pdx.datacollector")
        self.config = config
        self.opcua_client = AsyncOPCUAClient(
            endpoint=config["opcua"]["endpoint"],
            anon_login=config["opcua"]["anon_login"],
        )
        self.nidaq_client = AsyncNIDAQClient(
            chassis=config["nidaq"]["chassis"],
            modules_config=config["nidaq"]["modules"],
        )
        self.start_trigger = asyncio.Event()
        self.stop_trigger = asyncio.Event()
        self._monitor_task: Optional[asyncio.Task] = None
        self._collection_task: Optional[asyncio.Task] = None
        self._cleanup_required = False
        self.collection_start_time: Optional[datetime] = None
        self.nidaq_indices: Optional[Dict[str, int]] = None

        # Set start trigger if not configured
        if not config.get("trigger", {}).get("start", {}).get("conditions"):
            self.start_trigger.set()

    async def verify_connections(self) -> bool:
        """Verify connections to both data sources"""
        try:
            # Verify OPCUA connection
            if not await self.opcua_client.connect():
                self.logger.error("Failed to connect to OPCUA server")
                return False

            # Verify NI-DAQ connection and configuration
            if not await self.nidaq_client.setup():
                self.logger.error("Failed to setup NI-DAQ")
                await self.opcua_client.disconnect()
                return False

            self.logger.info("Successfully connected to all data sources")
            return True

        except Exception as e:
            self.logger.error(f"Error verifying connections: {e}")
            return False

    async def monitor_start_trigger(self):
        """Monitor start trigger conditions"""
        try:
            while not self.start_trigger.is_set():
                # Check start trigger conditions
                start_conditions = self.config["trigger"]["start"]["conditions"]
                start_logic = self.config["trigger"]["start"]["logic"]

                if start_logic == "and":
                    if all(
                        [await self._check_condition(cond) for cond in start_conditions]
                    ):
                        self.start_trigger.set()
                elif start_logic == "or":
                    if any(
                        [await self._check_condition(cond) for cond in start_conditions]
                    ):
                        self.start_trigger.set()
                else:
                    raise Exception(f"Unsupported logic condition {start_logic}!")
                await asyncio.sleep(0.1)
        except Exception as e:
            self.logger.error(f"Error in start trigger monitoring: {e}")
            self.logger.error("Beginning data collection immediately.")
            self.start_trigger.set()

    async def monitor_stop_trigger(self):
        """Monitor stop trigger conditions"""
        try:
            if "stop" in self.config.get("trigger", {}):
                while not self.stop_trigger.is_set():
                    # Check stop trigger conditions
                    stop_conditions = self.config["trigger"]["stop"]["conditions"]
                    stop_logic = self.config["trigger"]["stop"]["logic"]

                    if stop_logic == "and":
                        if all(
                            [
                                await self._check_condition(cond)
                                for cond in stop_conditions
                            ]
                        ):
                            self.stop_trigger.set()
                    elif stop_logic == "or":
                        if any(
                            [
                                await self._check_condition(cond)
                                for cond in stop_conditions
                            ]
                        ):
                            self.stop_trigger.set()
                    await asyncio.sleep(0.1)
        except Exception as e:
            self.logger.error(f"Error in stop trigger monitoring: {e}")
            self.logger.error("Ending data collection immediately.")
            self.stop_trigger.set()

    async def _check_condition(self, condition: dict) -> bool:
        """Check a single trigger condition"""
        # TODO: only opcua triggers accepted for now
        try:
            node_id = [
                k
                for k, v in self.config["opcua"]["node_info"].items()
                if v["label"] == condition["node"]
            ]
            if node_id:
                value = await self.opcua_client.read_node_value(node_id[0])
                operator = condition["operator"]
                threshold = condition["value"]

                if operator == "==":
                    return value == threshold
                elif operator == "!=":
                    return value != threshold
                elif operator == ">":
                    return value > threshold
                elif operator == ">=":
                    return value >= threshold
                elif operator == "<":
                    return value < threshold
                elif operator == "<=":
                    return value <= threshold
                else:
                    self.logger.error(f"Unknown operator: {operator}")
                    return False
            else:
                return False

        except Exception as e:
            self.logger.error(f"Error checking condition: {e}")
            return False

    async def nidaq_data_callback(self, nidaq_data: Dict[str, np.ndarray]):
        """Callback to handle NI-DAQ data chunks."""
        if self.nidaq_group is None or self.nidaq_indices is None:
            self.logger.warning("HDF5 file not ready for NI-DAQ data, skipping chunk.")
            return

        for slot_id, module_data in nidaq_data.items():
            if slot_id not in self.nidaq_client.modules:
                continue

            module_group = cast(Group, self.nidaq_group[slot_id])
            chunk_size = module_data.shape[1]
            current_index = self.nidaq_indices[slot_id]
            new_index = current_index + chunk_size

            # Check if we would write past the allocated size
            if new_index > cast(Dataset, module_group["timestamp"]).shape[0]:
                self.logger.warning(
                    f"NI-DAQ buffer for {slot_id} is full. Trimming will occur after collection."
                )
                continue

            # Update timestamps
            timestamps = np.linspace(
                current_index / self.nidaq_client.modules[slot_id].sample_rate,
                new_index / self.nidaq_client.modules[slot_id].sample_rate,
                chunk_size,
                endpoint=False,  # Important for continuous time vector
            )

            cast(Dataset, module_group["timestamp"])[current_index:new_index] = (
                timestamps
            )

            # Store channel data
            for i, (_, channel_config) in enumerate(
                self.nidaq_client.modules[slot_id].channels.items()
            ):
                dataset = cast(Dataset, module_group[channel_config.name])
                dataset[current_index:new_index] = module_data[i]

            self.nidaq_indices[slot_id] = new_index

    async def collect_data(self, output_file: Path):
        """Main data collection routine"""
        try:
            # Verify connections first
            if not await self.verify_connections():
                return False

            self._cleanup_required = True

            # Start trigger monitoring only if trigger conditions exist
            if not self.start_trigger.is_set():
                self.logger.info("Waiting for start trigger conditions...")
                self._monitor_task = asyncio.create_task(self.monitor_start_trigger())
                try:
                    await self.start_trigger.wait()
                except asyncio.CancelledError:
                    self.logger.info("Start trigger monitoring cancelled.")
                    return False
            else:
                self.logger.info(
                    "No start trigger configured, beginning collection immediately"
                )

            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)

            self.collection_start_time = datetime.now(timezone.utc)

            # Start NI-DAQ acquisition
            if not await self.nidaq_client.start_acquisition():
                self.logger.error("Failed to start NI-DAQ acquisition")
                return False

            with h5py.File(output_file, "w") as f:
                self.h5_file = f
                # Create metadata group
                meta_group: Group = f.create_group("metadata")
                meta_group.attrs["start_time"] = self.collection_start_time.isoformat()
                meta_group.attrs["asset"] = self.config["asset"]

                # Create OPCUA dataset
                opcua_group: Group = f.create_group("opcua_data")
                opcua_group.create_dataset(
                    "timestamp", shape=(0,), maxshape=(None,), dtype="float64"
                )
                for node_id, node_info in self.config["opcua"]["node_info"].items():
                    dtype = node_info.get("dtype", "float64")
                    if dtype == "string":
                        dtype = h5py.string_dtype(encoding="utf-8")
                    opcua_group.create_dataset(
                        node_info["label"], shape=(0,), maxshape=(None,), dtype=dtype
                    )

                # Setup NI-DAQ datasets for each module
                self.nidaq_group = f.create_group("nidaq_data")
                self.nidaq_indices = {
                    slot_id: 0 for slot_id in self.nidaq_client.modules.keys()
                }

                for slot_id, module in self.nidaq_client.modules.items():
                    module_group: Group = self.nidaq_group.create_group(slot_id)
                    chunk_size = module.samples_per_chunk
                    estimated_chunks = int(
                        self.config["duration"] / module.chunk_duration * 1.5
                    )
                    max_samples = chunk_size * estimated_chunks

                    module_group.create_dataset(
                        "timestamp",
                        shape=(max_samples,),
                        maxshape=(max_samples,),
                        dtype="float64",
                        chunks=(chunk_size,),
                    )
                    for _, channel_config in module.channels.items():
                        module_group.create_dataset(
                            channel_config.name,
                            shape=(max_samples,),
                            maxshape=(max_samples,),
                            dtype="float64",
                            chunks=(chunk_size,),
                        )

                async def collect_opcua():
                    """Separate task for OPC UA data collection"""
                    start_time = self.collection_start_time
                    if start_time is None:
                        self.logger.error(
                            "collect_opcua called before collection_start_time was set."
                        )
                        return
                    opcua_index = 0
                    with alive_bar() as bar:
                        while (
                            not self.stop_trigger.is_set()
                            and (get_utcnow() - start_time).total_seconds()
                            < self.config["duration"]
                        ):
                            opcua_data = await self.opcua_client.collect_data(
                                self.config["opcua"]["node_info"]
                            )
                            new_size = opcua_index + 1
                            cast(Dataset, opcua_group["timestamp"]).resize((new_size,))
                            relative_time = (
                                datetime.fromisoformat(opcua_data["AbsTimestamp"])
                                - start_time
                            ).total_seconds()
                            cast(Dataset, opcua_group["timestamp"])[opcua_index] = (
                                relative_time
                            )
                            for node_info in self.config["opcua"]["node_info"].values():
                                node_label = node_info["label"]
                                dataset = cast(Dataset, opcua_group[node_label])
                                dataset.resize((new_size,))
                                value = opcua_data.get(node_label)
                                if isinstance(value, datetime):
                                    dataset[opcua_index] = value.timestamp()
                                else:
                                    dataset[opcua_index] = value
                            opcua_index += 1
                            await asyncio.sleep(self.config["opcua"]["interval"])
                            bar()

                self.logger.info(
                    f"Collecting data for up to {self.config['duration']} seconds or until stop trigger..."
                )
                stop_monitor_task = asyncio.create_task(self.monitor_stop_trigger())
                opcua_task = asyncio.create_task(collect_opcua())
                nidaq_task = asyncio.create_task(
                    self.nidaq_client.run_acquisition(self.nidaq_data_callback)
                )

                collection_tasks = [opcua_task, nidaq_task]
                done, pending = await asyncio.wait(
                    collection_tasks + [stop_monitor_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if stop_monitor_task in done:
                    self.logger.info("Stop trigger condition met. Stopping collection.")

                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

                meta_group.attrs["end_time"] = datetime.now(timezone.utc).isoformat()

                if self.nidaq_indices and self.nidaq_group:
                    self.logger.info("Trimming excess NI-DAQ data...")
                    for slot_id, final_index in self.nidaq_indices.items():
                        module_group = cast(Group, self.nidaq_group[slot_id])
                        self.logger.debug(
                            f"Trimming module {slot_id} data to {final_index} samples."
                        )
                        cast(Dataset, module_group["timestamp"]).resize((final_index,))
                        for channel_config in self.nidaq_client.modules[
                            slot_id
                        ].channels.values():
                            cast(Dataset, module_group[channel_config.name]).resize(
                                (final_index,)
                            )
                    self.logger.info("Finished trimming data.")
            return True
        except Exception as e:
            self.logger.error(f"Error in data collection: {e}", exc_info=True)
            return False
        finally:
            if self._cleanup_required:
                await self.cleanup()

    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self._monitor_task:
                self._monitor_task.cancel()
                try:
                    await self._monitor_task
                except asyncio.CancelledError:
                    pass

            await self.nidaq_client.cleanup()
            await self.opcua_client.disconnect()

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
