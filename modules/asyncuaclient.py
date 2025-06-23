import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Optional, Any
from asyncua import Client, Node


class AsyncOPCUAClient:
    def __init__(self, endpoint: str, anon_login: bool = False):
        self.logger = logging.getLogger("pdx.asyncuaclient")
        self.endpoint = endpoint
        self.client = Client(url=endpoint)
        if not anon_login:
            self.client.set_user(os.getenv("OPCUA_USER", ""))
            self.client.set_password(os.getenv("OPCUA_PASS", ""))
        self._nodes: Dict[str, Node] = {}
        self._connected = False
        self._subscription = None
        self._last_values: Dict[str, Any] = {}
        self._connection_retry_count = 0
        self._max_retries = 3

    async def get_node(self, node_id: str) -> Optional[Node]:
        """
        Get node with caching to avoid repeated lookups
        """
        if node_id not in self._nodes:
            try:
                self._nodes[node_id] = self.client.get_node(node_id)
            except Exception as e:
                self.logger.error(f"Error getting node {node_id}: {e}")
                return None
        return self._nodes[node_id]

    async def read_node_value(self, node_id: str) -> Optional[Any]:
        """
        Read value from node with error handling
        """
        try:
            node = await self.get_node(node_id)
            if node:
                value = await node.read_value()
                self._last_values[node_id] = value
                return value
            return None
        except Exception as e:
            self.logger.error(f"Error reading value from node {node_id}: {e}")
            return self._last_values.get(node_id)

    async def connect(self) -> bool:
        """
        Connect to the OPC UA server with retry logic
        """
        while self._connection_retry_count < self._max_retries:
            try:
                await self.client.connect()
                self._connected = True
                self.logger.info(f"Connected to OPC UA server at {self.endpoint}")
                return True
            except Exception as e:
                self._connection_retry_count += 1
                self.logger.error(
                    f"Connection attempt {self._connection_retry_count} failed: {e}"
                )
                if self._connection_retry_count < self._max_retries:
                    await asyncio.sleep(1)
        return False

    async def disconnect(self):
        """
        Disconnect from the OPC UA server
        """
        if self._connected:
            try:
                await self.client.disconnect()
                self._connected = False
                self.logger.info("Disconnected from OPC UA server")
            except Exception as e:
                self.logger.error(f"Error during disconnect: {e}")

    async def collect_data(
        self, nodes: Optional[Dict[str, Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Collect data from specified nodes
        """
        if not self._connected:
            if not await self.connect():
                return {}

        data: Dict[str, Any] = {"AbsTimestamp": datetime.now(timezone.utc).isoformat()}

        if nodes:
            nodes_to_read = nodes
        else:
            # Fallback to all cached nodes, using node_id as label
            nodes_to_read = {node_id: {"label": node_id} for node_id in self._nodes}

        if not nodes_to_read:
            return data

        read_tasks = [self.read_node_value(node_id) for node_id in nodes_to_read.keys()]

        try:
            values = await asyncio.gather(*read_tasks, return_exceptions=True)

            for (node_id, node_info), value in zip(nodes_to_read.items(), values):
                node_label = node_info["label"]
                if isinstance(value, Exception):
                    self.logger.error(f"Error reading {node_label}: {value}")
                    data[node_label] = self._last_values.get(node_id)
                else:
                    data[node_label] = value
                    self._last_values[node_id] = value

        except Exception as e:
            self.logger.error(f"Error collecting data: {e}")
            for node_id, node_info in nodes_to_read.items():
                node_label = node_info["label"]
                data[node_label] = self._last_values.get(node_id)

        return data

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
