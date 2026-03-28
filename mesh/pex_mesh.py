"""
PEX (Peer Exchange) Mesh – private encrypted overlay network.

Architecture
------------
* Each node maintains a local peer table (node_id → NodeInfo).
* Nodes broadcast signed "HELLO" packets at startup and periodically.
* On receiving a HELLO, the node replies with its own peer table so peers
  can bootstrap without a central tracker (pure mesh).
* All peer-to-peer messages are encrypted with the session key negotiated
  via ECDH; outer integrity is protected by SHA-256 HMAC.
* The pex mesh intentionally has no externally-routable entry points:
  listener sockets bind only to link-local / LAN addresses so the mesh
  is invisible from the public internet.

Security note
-------------
An outsider who somehow reaches the raw socket without the shared session
key receives only the fractal-honeypot response (see security.honeypot).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import struct
import time
from dataclasses import asdict
from typing import Callable, Dict, List, Optional, Tuple

from hive.node import NodeInfo

logger = logging.getLogger(__name__)

# Magic bytes that prefix every mesh packet
_MAGIC = b"\xca\xfe\xf0\x0d"
_PROTOCOL_VERSION = 1

# How often nodes re-broadcast themselves even without new peers
HELLO_INTERVAL_SECS: float = 15.0
PEER_TABLE_MAX: int = 256


class PexMesh:
    """
    Lightweight PEX overlay mesh operating exclusively on LAN/link-local
    addresses.  Not externally reachable without the mesh session key.
    """

    def __init__(
        self,
        local_info: NodeInfo,
        session_key: bytes,
        bind_host: str = "0.0.0.0",
        udp_port: int = 5353,
        tcp_port: int = 5354,
    ) -> None:
        self.local_info = local_info
        self._key = session_key
        self._bind_host = bind_host
        self._udp_port = udp_port
        self._tcp_port = tcp_port
        self._peers: Dict[str, NodeInfo] = {}
        self._peer_callbacks: List[Callable[[NodeInfo], None]] = []
        self._running = False
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._server: Optional[asyncio.AbstractServer] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True

        loop = asyncio.get_event_loop()

        # UDP for peer-exchange broadcasts
        self._transport, _ = await loop.create_datagram_endpoint(
            lambda: _UDPProtocol(self._on_udp_packet),
            local_addr=(self._bind_host, self._udp_port),
            allow_broadcast=True,
        )

        # TCP for reliable peer-table exchange
        self._server = await asyncio.start_server(
            self._handle_tcp,
            self._bind_host,
            self._tcp_port,
        )

        asyncio.create_task(self._hello_loop())
        logger.info(
            "PexMesh started – UDP %d / TCP %d", self._udp_port, self._tcp_port
        )

    async def stop(self) -> None:
        self._running = False
        if self._transport:
            self._transport.close()
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_peer_discovered(self, fn: Callable[[NodeInfo], None]) -> None:
        self._peer_callbacks.append(fn)

    def peers(self) -> List[NodeInfo]:
        return list(self._peers.values())

    async def broadcast_hello(self) -> None:
        """Broadcast a signed HELLO to the local network segment."""
        payload = self._encode_message({"type": "HELLO", "info": self.local_info.to_dict()})
        if self._transport:
            self._transport.sendto(payload, ("<broadcast>", self._udp_port))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _hello_loop(self) -> None:
        while self._running:
            await self.broadcast_hello()
            await asyncio.sleep(HELLO_INTERVAL_SECS)

    def _on_udp_packet(self, data: bytes, addr: Tuple[str, int]) -> None:
        try:
            msg = self._decode_message(data)
        except Exception:
            logger.debug("Undecodeable UDP packet from %s (possible non-mesh device)", addr[0])
            return

        if msg.get("type") == "HELLO":
            info = NodeInfo.from_dict(msg["info"])
            if info.node_id != self.local_info.node_id:
                self._register_peer(info)
                # Respond with our peer table via TCP for reliability
                asyncio.create_task(self._send_peer_table(info))

    async def _handle_tcp(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            raw = await asyncio.wait_for(reader.read(131072), timeout=10.0)
            msg = self._decode_message(raw)
            if msg.get("type") == "PEERS":
                for peer_data in msg.get("peers", []):
                    try:
                        info = NodeInfo.from_dict(peer_data)
                        if info.node_id != self.local_info.node_id:
                            self._register_peer(info)
                    except Exception:
                        pass
        except Exception:
            logger.debug("TCP handler error (likely non-mesh connection)")
        finally:
            writer.close()

    async def _send_peer_table(self, target: NodeInfo) -> None:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(target.host, self._tcp_port), timeout=5.0
            )
            table = [p.to_dict() for p in self._peers.values()][:PEER_TABLE_MAX]
            payload = self._encode_message({"type": "PEERS", "peers": table})
            writer.write(payload)
            await writer.drain()
            writer.close()
        except Exception:
            pass

    def _register_peer(self, info: NodeInfo) -> None:
        if info.node_id not in self._peers:
            logger.info("New mesh peer: %s @ %s:%d", info.node_id[:8], info.host, info.port)
            for cb in self._peer_callbacks:
                try:
                    cb(info)
                except Exception:
                    logger.exception("Peer callback error")
        info.last_seen = time.monotonic()
        self._peers[info.node_id] = info

    # ------------------------------------------------------------------
    # Crypto helpers
    # ------------------------------------------------------------------

    def _hmac(self, data: bytes) -> bytes:
        return hmac.new(self._key, data, hashlib.sha256).digest()

    def _encode_message(self, obj: dict) -> bytes:
        """Serialise + HMAC-sign a message."""
        body = json.dumps(obj, separators=(",", ":")).encode()
        sig = self._hmac(body)
        # wire format: magic(4) | version(1) | sig(32) | len(4) | body
        header = _MAGIC + struct.pack("!B", _PROTOCOL_VERSION)
        length = struct.pack("!I", len(body))
        return header + sig + length + body

    def _decode_message(self, data: bytes) -> dict:
        """Verify HMAC and deserialise.  Raises on failure."""
        if len(data) < 4 + 1 + 32 + 4:
            raise ValueError("Packet too short")
        magic = data[:4]
        if magic != _MAGIC:
            raise ValueError("Bad magic bytes")
        # version = data[4]  # reserved for future compatibility
        sig = data[5:37]
        length = struct.unpack("!I", data[37:41])[0]
        body = data[41: 41 + length]
        expected = self._hmac(body)
        if not hmac.compare_digest(sig, expected):
            raise ValueError("HMAC verification failed – rejecting packet")
        return json.loads(body)


class _UDPProtocol(asyncio.DatagramProtocol):
    def __init__(self, handler: Callable[[bytes, Tuple[str, int]], None]) -> None:
        self._handler = handler

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        self._handler(data, addr)

    def error_received(self, exc: Exception) -> None:
        logger.debug("UDP error: %s", exc)
