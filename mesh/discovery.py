"""
NodeDiscovery – mDNS/Bonjour service discovery for hive nodes on the LAN.

When a node starts it registers itself under the service type
``_jessicai-hive._tcp.local.`` so that any other node on the same L2
segment can find it without prior configuration.
"""

from __future__ import annotations

import asyncio
import logging
import socket
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

_SERVICE_TYPE = "_jessicai-hive._tcp.local."


class NodeDiscovery:
    """
    Wraps the ``zeroconf`` library for zero-configuration node discovery.

    Discovered nodes are forwarded to registered callbacks so the mesh
    and cluster layers can act on them immediately.
    """

    def __init__(self, node_id: str, host: str, port: int) -> None:
        self._node_id = node_id
        self._host = host
        self._port = port
        self._callbacks: List[Callable[[str, str, int], None]] = []
        self._zc: Optional[object] = None
        self._service_info: Optional[object] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        await asyncio.get_event_loop().run_in_executor(None, self._start_sync)

    async def stop(self) -> None:
        await asyncio.get_event_loop().run_in_executor(None, self._stop_sync)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def on_node_found(self, fn: Callable[[str, str, int], None]) -> None:
        """Register a callback invoked as fn(node_id, host, port)."""
        self._callbacks.append(fn)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _start_sync(self) -> None:
        try:
            from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf

            local_addr = self._resolve_host()
            self._zc = Zeroconf(interfaces=[local_addr])

            self._service_info = ServiceInfo(
                _SERVICE_TYPE,
                f"{self._node_id}.{_SERVICE_TYPE}",
                addresses=[socket.inet_aton(local_addr)],
                port=self._port,
                properties={b"id": self._node_id.encode()},
            )
            self._zc.register_service(self._service_info)

            ServiceBrowser(self._zc, _SERVICE_TYPE, handlers=[self._on_service_state_change])
            logger.info("NodeDiscovery started – advertising %s:%d", local_addr, self._port)
        except ImportError:
            logger.warning(
                "zeroconf not installed – mDNS discovery unavailable.  "
                "Install it with: pip install zeroconf"
            )
        except Exception:
            logger.exception("NodeDiscovery._start_sync error")

    def _stop_sync(self) -> None:
        try:
            if self._zc:
                if self._service_info:
                    self._zc.unregister_service(self._service_info)
                self._zc.close()
        except Exception:
            logger.exception("NodeDiscovery._stop_sync error")

    def _on_service_state_change(self, zeroconf, service_type, name, state_change) -> None:
        try:
            from zeroconf import ServiceStateChange
            if state_change is not ServiceStateChange.Added:
                return
            info = zeroconf.get_service_info(service_type, name)
            if info is None:
                return
            host = socket.inet_ntoa(info.addresses[0])
            port = info.port
            node_id = info.properties.get(b"id", b"").decode()
            if node_id and node_id != self._node_id:
                logger.info("Discovered peer via mDNS: %s @ %s:%d", node_id[:8], host, port)
                for cb in self._callbacks:
                    try:
                        cb(node_id, host, port)
                    except Exception:
                        logger.exception("mDNS callback error")
        except Exception:
            logger.exception("Service state change handler error")

    def _resolve_host(self) -> str:
        if self._host not in ("0.0.0.0", ""):
            return self._host
        # Auto-detect the first non-loopback IPv4 address
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            addr = s.getsockname()[0]
            s.close()
            return addr
        except Exception:
            return "127.0.0.1"
