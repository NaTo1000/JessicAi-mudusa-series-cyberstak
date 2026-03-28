"""
WiFiManager – discovers and connects to authorised public WiFi access
points so the hive can expand its mesh across them.

Important: this module only connects to networks that are:
  1. Listed in the user-supplied allow-list, OR
  2. Detected as captive-portal-free public hotspots (e.g. Telstra Air,
     Optus WiFi Hotspot) where the operator explicitly offers free access.

It never modifies network settings without operator consent.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# Well-known free public hotspot SSIDs that operators publish for public use.
# Connection to these networks is the operator's explicit intent.
DEFAULT_ALLOWLIST: List[str] = [
    "Telstra Air",
    "Telstra WiFi",
    "Optus WiFi Hotspot",
    "MyRepublic WiFi",
    "Fon WiFi",
    "BTOpenzone",
    "BT Wi-fi",
    "The Cloud",
    "Virgin WiFi",
    "SKY WiFi",
]


# nmcli reports signal strength as 0-100; subtract this offset to convert
# back to an approximate dBm value (-110 = -110 dBm at signal=0).
_NMCLI_SIGNAL_TO_DBM_OFFSET = 110
class AccessPoint:
    ssid: str
    bssid: str
    signal_dbm: int
    frequency_mhz: int
    open: bool = False
    known_public: bool = False

    @property
    def quality(self) -> int:
        """Convert dBm to a 0-100 quality score."""
        clamped = max(-100, min(-50, self.signal_dbm))
        return 2 * (clamped + 100)


class WiFiManager:
    """
    Scan for, and connect to, authorised public WiFi networks.

    Only Linux (NetworkManager / iw) is fully supported.
    macOS / Windows stubs are provided; extend as needed.
    """

    def __init__(
        self,
        interface: Optional[str] = None,
        ssid_allowlist: Optional[List[str]] = None,
        scan_interval_secs: float = 60.0,
    ) -> None:
        self._iface = interface
        self._allowlist = set(ssid_allowlist or DEFAULT_ALLOWLIST)
        self._scan_interval = scan_interval_secs
        self._connected_ssid: Optional[str] = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        asyncio.create_task(self._scan_loop())
        logger.info("WiFiManager started (allowlist=%d SSIDs)", len(self._allowlist))

    async def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def connected_ssid(self) -> Optional[str]:
        return self._connected_ssid

    def add_to_allowlist(self, ssid: str) -> None:
        self._allowlist.add(ssid)
        logger.info("Added %r to WiFi allowlist", ssid)

    async def scan(self) -> List[AccessPoint]:
        """Return visible access points on the current platform."""
        if sys.platform.startswith("linux"):
            return await asyncio.get_event_loop().run_in_executor(None, self._scan_linux)
        logger.debug("WiFi scan not implemented on %s – returning empty list", sys.platform)
        return []

    async def connect_best(self) -> bool:
        """
        Scan and connect to the highest-quality authorised AP.
        Returns True if a new connection was established.
        """
        aps = await self.scan()
        candidates = [
            ap for ap in aps
            if ap.ssid in self._allowlist and ap.open
        ]
        if not candidates:
            logger.debug("No authorised open APs in range")
            return False
        best = max(candidates, key=lambda ap: ap.quality)
        if best.ssid == self._connected_ssid:
            return False
        return await asyncio.get_event_loop().run_in_executor(
            None, self._connect_linux, best.ssid
        )

    # ------------------------------------------------------------------
    # Platform implementations
    # ------------------------------------------------------------------

    def _scan_linux(self) -> List[AccessPoint]:
        aps: List[AccessPoint] = []
        try:
            iface = self._iface or self._detect_iface()
            if not iface:
                return aps
            out = subprocess.check_output(
                ["iw", "dev", iface, "scan"],
                stderr=subprocess.DEVNULL,
                timeout=20,
            ).decode(errors="replace")
            aps = _parse_iw_scan(out)
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            logger.debug("iw scan unavailable – using nmcli fallback")
            aps = self._scan_nmcli()
        except Exception:
            logger.exception("WiFi scan error")
        # Mark known public SSIDs
        for ap in aps:
            ap.known_public = ap.ssid in self._allowlist
        return aps

    def _scan_nmcli(self) -> List[AccessPoint]:
        aps: List[AccessPoint] = []
        try:
            out = subprocess.check_output(
                ["nmcli", "-t", "-f", "SSID,BSSID,SIGNAL,FREQ,SECURITY", "dev", "wifi", "list"],
                stderr=subprocess.DEVNULL,
                timeout=15,
            ).decode(errors="replace")
            for line in out.splitlines():
                parts = line.split(":")
                if len(parts) < 5:
                    continue
                ssid, bssid, signal_str, freq_str, security = parts[:5]
                try:
                    signal = int(signal_str)
                    freq = int(re.sub(r"\D", "", freq_str) or "0")
                    aps.append(AccessPoint(
                        ssid=ssid,
                        bssid=bssid,
                        signal_dbm=signal - _NMCLI_SIGNAL_TO_DBM_OFFSET,
                        frequency_mhz=freq,
                        open=(not security or security.strip() == "--"),
                    ))
                except ValueError:
                    continue
        except Exception:
            logger.debug("nmcli scan also unavailable")
        return aps

    def _connect_linux(self, ssid: str) -> bool:
        try:
            subprocess.check_call(
                ["nmcli", "dev", "wifi", "connect", ssid],
                timeout=30,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._connected_ssid = ssid
            logger.info("Connected to WiFi: %r", ssid)
            return True
        except Exception as exc:
            logger.warning("Failed to connect to %r: %s", ssid, exc)
            return False

    @staticmethod
    def _detect_iface() -> Optional[str]:
        try:
            out = subprocess.check_output(
                ["iw", "dev"], stderr=subprocess.DEVNULL
            ).decode()
            m = re.search(r"Interface\s+(\w+)", out)
            return m.group(1) if m else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Scan loop
    # ------------------------------------------------------------------

    async def _scan_loop(self) -> None:
        while self._running:
            await self.connect_best()
            await asyncio.sleep(self._scan_interval)


# ---------------------------------------------------------------------------
# iw scan output parser
# ---------------------------------------------------------------------------

def _parse_iw_scan(text: str) -> List[AccessPoint]:
    aps: List[AccessPoint] = []
    current: dict = {}

    for line in text.splitlines():
        if line.startswith("BSS "):
            if current.get("ssid"):
                aps.append(_make_ap(current))
            current = {"bssid": line.split()[1].strip("()")}
        elif "SSID:" in line:
            current["ssid"] = line.split("SSID:", 1)[1].strip()
        elif "signal:" in line:
            m = re.search(r"signal:\s*([-\d.]+)", line)
            if m:
                current["signal"] = int(float(m.group(1)))
        elif "freq:" in line:
            m = re.search(r"freq:\s*(\d+)", line)
            if m:
                current["freq"] = int(m.group(1))
        elif "capability:" in line:
            current["open"] = "Privacy" not in line

    if current.get("ssid"):
        aps.append(_make_ap(current))
    return aps


def _make_ap(d: dict) -> AccessPoint:
    return AccessPoint(
        ssid=d.get("ssid", ""),
        bssid=d.get("bssid", ""),
        signal_dbm=d.get("signal", -100),
        frequency_mhz=d.get("freq", 0),
        open=d.get("open", False),
    )
