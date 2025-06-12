import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any
import h5py
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

    async def monitor_triggers(self):
        """Monitor start and stop trigger conditions"""
        try:
            while not self.start_trigger.is_set() or not self.stop_trigger.is_set():
                if not self.start_trigger.is_set():
                    # Check start trigger conditions
                    start_conditions = self.config["trigger"]["start"]["conditions"]
                    start_logic = self.config["trigger"]["start"]["logic"]

                    if start_logic == "and":
                        if all(
                            [
                                await self._check_condition(cond)
                                for cond in start_conditions
                            ]
                        ):
                            self.start_trigger.set()
                    else:  # "or"
                        if any(
                            [
                                await self._check_condition(cond)
                                for cond in start_conditions
                            ]
                        ):
                            self.start_trigger.set()

                if not self.stop_trigger.is_set():
                    # Check stop trigger conditions if configured
                    if "stop" in self.config.get("trigger", {}):
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
                        else:  # "or"
                            if any(
                                [
                                    await self._check_condition(cond)
                                    for cond in stop_conditions
                                ]
                            ):
                                self.stop_trigger.set()

                await asyncio.sleep(0.1)

        except Exception as e:
            self.logger.error(f"Error in trigger monitoring: {e}")
            self.start_trigger.set()
            self.stop_trigger.set()

    async def _check_condition(self, condition: dict) -> bool:
        """Check a single trigger condition"""
        try:
            value = await self.opcua_client.read_node_value(condition["node"])
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

        except Exception as e:
            self.logger.error(f"Error checking condition: {e}")
            return False

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
                self._monitor_task = asyncio.create_task(self.monitor_triggers())

                try:
                    await asyncio.wait_for(self.start_trigger.wait(), timeout=None)
                except asyncio.TimeoutError:
                    self.logger.error("Timeout waiting for start trigger")
                    return False
            else:
                self.logger.info(
                    "No start trigger configured, beginning collection immediately"
                )

            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)

            collection_start_time = datetime.now(timezone.utc)

            # Start NI-DAQ acquisition
            if not await self.nidaq_client.start_acquisition():
                self.logger.error("Failed to start NI-DAQ acquisition")
                return False

            with h5py.File(output_file, "w") as f:
                # Create metadata group
                meta_group = f.create_group("metadata")
                meta_group.attrs["start_time"] = collection_start_time.isoformat()
                meta_group.attrs["asset"] = self.config["asset"]

                # Create OPCUA dataset
                opcua_group = f.create_group("opcua_data")
                opcua_group.create_dataset(
                    "timestamp", shape=(0,), maxshape=(None,), dtype="float64"
                )

                for node_id, node_label in self.config["opcua"]["node_info"].items():
                    opcua_group.create_dataset(
                        node_label, shape=(0,), maxshape=(None,), dtype="float64"
                    )

                # Setup NI-DAQ datasets for each module
                nidaq_group = f.create_group("nidaq_data")

                # Initialize indices dictionary for each module
                nidaq_indices = {
                    slot_id: 0 for slot_id in self.nidaq_client.modules.keys()
                }

                for slot_id, module in self.nidaq_client.modules.items():
                    module_group = nidaq_group.create_group(slot_id)
                    chunk_size = module.samples_per_chunk
                    # Add safety factor to estimated chunks (e.g., 1.5x)
                    estimated_chunks = int(
                        self.config["duration"] / module.chunk_duration * 1.5
                    )
                    max_samples = chunk_size * estimated_chunks

                    # Create timestamp dataset for this module
                    module_group.create_dataset(
                        "timestamp",
                        shape=(max_samples,),
                        maxshape=(max_samples,),
                        dtype="float64",
                        chunks=(chunk_size,),
                    )

                    # Create datasets for each channel in this module
                    for channel_id, channel_config in module.channels.items():
                        module_group.create_dataset(
                            channel_config.name,
                            shape=(max_samples,),
                            maxshape=(max_samples,),
                            dtype="float64",
                            chunks=(chunk_size,),
                        )

                async def collect_nidaq():
                    """Separate task for NI-DAQ data collection"""
                    while (
                        get_utcnow() - collection_start_time
                    ).total_seconds() < self.config["duration"]:
                        nidaq_data = await self.nidaq_client.read_chunk()
                        for slot_id, module_data in nidaq_data.items():
                            module_group = nidaq_group[slot_id]
                            chunk_size = module_data.shape[1]
                            current_index = nidaq_indices[slot_id]
                            new_index = current_index + chunk_size

                            # Update timestamps
                            timestamps = np.linspace(
                                current_index
                                / self.nidaq_client.modules[slot_id].sample_rate,
                                new_index
                                / self.nidaq_client.modules[slot_id].sample_rate,
                                chunk_size,
                            )

                            module_group["timestamp"][current_index:new_index] = (
                                timestamps
                            )

                            # Store channel data
                            for i, (_, channel_config) in enumerate(
                                self.nidaq_client.modules[slot_id].channels.items()
                            ):
                                dataset = module_group[channel_config.name]
                                dataset[current_index:new_index] = module_data[i]

                            nidaq_indices[slot_id] = new_index

                async def collect_opcua():
                    """Separate task for OPC UA data collection"""
                    opcua_index = 0
                    while (
                        get_utcnow() - collection_start_time
                    ).total_seconds() < self.config["duration"]:
                        opcua_data = await self.opcua_client.collect_data(
                            self.config["opcua"]["node_info"]
                        )

                        # Store OPCUA data
                        new_size = opcua_index + 1
                        opcua_group["timestamp"].resize((new_size,))
                        relative_time = (
                            datetime.fromisoformat(opcua_data["AbsTimestamp"])
                            - collection_start_time
                        ).total_seconds()
                        opcua_group["timestamp"][opcua_index] = relative_time

                        for node_id, node_label in self.config["opcua"][
                            "node_info"
                        ].items():
                            dataset = opcua_group[node_label]
                            dataset.resize((new_size,))
                            value = opcua_data.get(node_label)
                            if isinstance(value, datetime):
                                dataset[opcua_index] = value.timestamp()
                            else:
                                dataset[opcua_index] = value

                        opcua_index += 1

                        # Sleep only affects OPCUA collection
                        await asyncio.sleep(self.config["opcua"]["interval"])

                # Run both collection tasks concurrently
                await asyncio.gather(collect_nidaq(), collect_opcua())

                # Store end time in metadata
                meta_group.attrs["end_time"] = datetime.now().isoformat()

                return True

        except Exception as e:
            self.logger.error(f"Error in data collection: {e}")
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
