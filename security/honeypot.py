"""
FractalHoneypot – deception layer for unauthorised tunnel connections.

An attacker who reaches the mesh port without the correct session key
does not see an error.  Instead they receive a continuous stream of
evolving 3-D fractal imagery rendered live from the fractal engine.

This achieves two goals:
  1. Deception – the attacker cannot determine whether they have reached
     anything real.
  2. Intelligence – all honeypot sessions are logged (timestamp,
     source address, bytes sent) for forensic review.
"""

from __future__ import annotations

import asyncio
import io
import logging
import struct
import time
from typing import Optional

logger = logging.getLogger(__name__)

# How many fractal frames to stream before closing the honeypot session.
# Each frame is sent immediately after the previous one completes, so the
# attacker experiences a smooth, never-ending animation.
HONEYPOT_FRAME_LIMIT: int = 0        # 0 = unlimited (stream forever)
FRAME_DELAY_SECS: float = 0.033      # ≈ 30 fps


class FractalHoneypot:
    """
    Streams continuously-evolving fractal imagery to unauthorised callers.

    The honeypot is driven by the FractalEngine (fractal.engine) so that
    every frame is genuinely unique and computationally authentic.
    """

    def __init__(self, frame_delay: float = FRAME_DELAY_SECS) -> None:
        self._frame_delay = frame_delay
        self._sessions: int = 0
        self._total_bytes: int = 0

    @property
    def sessions_served(self) -> int:
        return self._sessions

    @property
    def total_bytes_sent(self) -> int:
        return self._total_bytes

    async def handle(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Serve fractal frames until the remote side disconnects."""
        peer = writer.get_extra_info("peername", ("unknown", 0))
        self._sessions += 1
        session_id = self._sessions
        start = time.monotonic()
        logger.warning(
            "[honeypot] Session #%d started from %s:%d",
            session_id, peer[0], peer[1],
        )

        try:
            from fractal.engine import FractalEngine
            engine = FractalEngine()
            frame_count = 0
            while True:
                if HONEYPOT_FRAME_LIMIT and frame_count >= HONEYPOT_FRAME_LIMIT:
                    break
                frame_bytes = engine.render_frame_bytes(frame_index=frame_count)
                # HTTP/chunked-style framing so a browser/curl can display it
                header = (
                    f"HTTP/1.1 200 OK\r\n"
                    f"Content-Type: image/png\r\n"
                    f"Content-Length: {len(frame_bytes)}\r\n"
                    f"X-Fractal-Frame: {frame_count}\r\n"
                    f"\r\n"
                ).encode()
                payload = header + frame_bytes
                writer.write(payload)
                await writer.drain()
                self._total_bytes += len(payload)
                frame_count += 1
                await asyncio.sleep(self._frame_delay)
        except ConnectionResetError:
            pass
        except Exception:
            logger.debug("[honeypot] Session #%d ended with error", session_id)
        finally:
            elapsed = time.monotonic() - start
            logger.warning(
                "[honeypot] Session #%d closed – duration=%.1fs bytes=%d",
                session_id, elapsed, self._total_bytes,
            )
            try:
                writer.close()
            except Exception:
                pass
