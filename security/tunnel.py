"""
EncryptedTunnel – transparent encrypted channel between two hive nodes.

Every byte sent through the tunnel is:
  1. Authenticated with HMAC-SHA256.
  2. Encrypted with NaCl secretbox / AES-256-GCM.
  3. Framed with a 4-byte length prefix.

Unauthorised connections (missing or wrong session key) see no error –
they are silently routed to the FractalHoneypot (security.honeypot).
"""

from __future__ import annotations

import asyncio
import logging
import struct
from collections.abc import Awaitable
from typing import Callable, Optional

from .encryption import MeshEncryption

logger = logging.getLogger(__name__)

_FRAME_HEADER_SIZE = 4  # uint32 big-endian


class EncryptedTunnel:
    """
    Async encrypted stream tunnel.

    Usage (server side)
    -------------------
    >>> server = await EncryptedTunnel.create_server(session_key, "0.0.0.0", 9000, handler)

    Usage (client side)
    -------------------
    >>> tunnel = await EncryptedTunnel.connect(session_key, "10.0.0.1", 9000)
    >>> await tunnel.send(b"hello")
    >>> data = await tunnel.recv()
    """

    def __init__(
        self,
        enc: MeshEncryption,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._enc = enc
        self._reader = reader
        self._writer = writer

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    async def connect(
        cls,
        session_key: bytes,
        host: str,
        port: int,
        timeout: float = 10.0,
    ) -> "EncryptedTunnel":
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
        return cls(MeshEncryption(session_key), reader, writer)

    @classmethod
    async def create_server(
        cls,
        session_key: bytes,
        bind_host: str,
        port: int,
        handler: Callable[["EncryptedTunnel"], Awaitable[None]],
        honeypot_handler: Optional[Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]] = None,
    ) -> asyncio.AbstractServer:
        enc = MeshEncryption(session_key)

        async def _accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            # Peek at the first frame; if it cannot be decrypted the caller
            # is not in possession of the session key → serve honeypot.
            try:
                raw_frame = await _recv_frame(reader)
                enc.decrypt(raw_frame)  # will raise if key is wrong
            except Exception:
                logger.info(
                    "Unauthenticated connection from %s → fractal honeypot",
                    writer.get_extra_info("peername"),
                )
                if honeypot_handler:
                    await honeypot_handler(reader, writer)
                else:
                    writer.close()
                return

            tunnel = cls(enc, reader, writer)
            await handler(tunnel)

        return await asyncio.start_server(_accept, bind_host, port)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    async def send(self, data: bytes) -> None:
        ct = self._enc.encrypt(data)
        frame = struct.pack("!I", len(ct)) + ct
        self._writer.write(frame)
        await self._writer.drain()

    async def recv(self, timeout: float = 30.0) -> bytes:
        raw = await asyncio.wait_for(_recv_frame(self._reader), timeout=timeout)
        return self._enc.decrypt(raw)

    def close(self) -> None:
        self._writer.close()


async def _recv_frame(reader: asyncio.StreamReader) -> bytes:
    header = await reader.readexactly(_FRAME_HEADER_SIZE)
    length = struct.unpack("!I", header)[0]
    return await reader.readexactly(length)
