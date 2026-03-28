"""Linux toolkit catalogue and runner for Kali-Linux and Black Arch tools."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class ToolCategory(Enum):
    """Security tool categories drawn from Kali-Linux and Black Arch."""

    INFORMATION_GATHERING = auto()
    VULNERABILITY_ANALYSIS = auto()
    WEB_APPLICATION = auto()
    DATABASE = auto()
    PASSWORD_ATTACKS = auto()
    WIRELESS = auto()
    REVERSE_ENGINEERING = auto()
    EXPLOITATION = auto()
    SNIFFING_SPOOFING = auto()
    POST_EXPLOITATION = auto()
    FORENSICS = auto()
    REPORTING = auto()
    SOCIAL_ENGINEERING = auto()
    MISCELLANEOUS = auto()


@dataclass
class ToolDescriptor:
    """Metadata for a single security tool."""

    name: str
    category: ToolCategory
    description: str
    package: str                  # apt / pacman package name
    distros: list[str] = field(default_factory=list)  # ["kali", "blackarch"]
    homepage: str = ""

    def is_installed(self) -> bool:
        return shutil.which(self.name) is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.name,
            "description": self.description,
            "package": self.package,
            "distros": self.distros,
            "homepage": self.homepage,
            "installed": self.is_installed(),
        }


# ---------------------------------------------------------------------------
# Tool catalogue (subset — representative tools from Kali and Black Arch)
# ---------------------------------------------------------------------------
_CATALOGUE: list[ToolDescriptor] = [
    # Information Gathering
    ToolDescriptor("nmap", ToolCategory.INFORMATION_GATHERING,
                   "Network exploration and security auditing",
                   "nmap", ["kali", "blackarch"], "https://nmap.org"),
    ToolDescriptor("masscan", ToolCategory.INFORMATION_GATHERING,
                   "TCP port scanner — transmits 10M packets/sec",
                   "masscan", ["kali", "blackarch"], "https://github.com/robertdavidgraham/masscan"),
    ToolDescriptor("recon-ng", ToolCategory.INFORMATION_GATHERING,
                   "Web reconnaissance framework",
                   "recon-ng", ["kali"], "https://github.com/lanmaster53/recon-ng"),
    ToolDescriptor("theharvester", ToolCategory.INFORMATION_GATHERING,
                   "Gather emails, names, subdomains from public sources",
                   "theharvester", ["kali", "blackarch"], "https://github.com/laramies/theHarvester"),
    ToolDescriptor("amass", ToolCategory.INFORMATION_GATHERING,
                   "In-depth attack surface mapping and asset discovery",
                   "amass", ["kali", "blackarch"], "https://github.com/owasp-amass/amass"),

    # Vulnerability Analysis
    ToolDescriptor("nikto", ToolCategory.VULNERABILITY_ANALYSIS,
                   "Web server scanner",
                   "nikto", ["kali", "blackarch"], "https://cirt.net/Nikto2"),
    ToolDescriptor("openvas", ToolCategory.VULNERABILITY_ANALYSIS,
                   "Full-featured vulnerability scanner",
                   "openvas", ["kali"], "https://www.openvas.org"),
    ToolDescriptor("lynis", ToolCategory.VULNERABILITY_ANALYSIS,
                   "Security auditing tool for Unix/Linux",
                   "lynis", ["kali", "blackarch"], "https://cisofy.com/lynis"),

    # Web Application
    ToolDescriptor("sqlmap", ToolCategory.WEB_APPLICATION,
                   "Automatic SQL injection detection and exploitation",
                   "sqlmap", ["kali", "blackarch"], "https://sqlmap.org"),
    ToolDescriptor("burpsuite", ToolCategory.WEB_APPLICATION,
                   "Web security testing platform",
                   "burpsuite", ["kali"], "https://portswigger.net/burp"),
    ToolDescriptor("gobuster", ToolCategory.WEB_APPLICATION,
                   "Directory / file and DNS busting tool",
                   "gobuster", ["kali", "blackarch"], "https://github.com/OJ/gobuster"),
    ToolDescriptor("ffuf", ToolCategory.WEB_APPLICATION,
                   "Fast web fuzzer",
                   "ffuf", ["kali", "blackarch"], "https://github.com/ffuf/ffuf"),

    # Password Attacks
    ToolDescriptor("hashcat", ToolCategory.PASSWORD_ATTACKS,
                   "Advanced password recovery utility",
                   "hashcat", ["kali", "blackarch"], "https://hashcat.net"),
    ToolDescriptor("john", ToolCategory.PASSWORD_ATTACKS,
                   "John the Ripper password cracker",
                   "john", ["kali", "blackarch"], "https://www.openwall.com/john"),
    ToolDescriptor("hydra", ToolCategory.PASSWORD_ATTACKS,
                   "Network logon cracker",
                   "hydra", ["kali", "blackarch"], "https://github.com/vanhauser-thc/thc-hydra"),

    # Wireless
    ToolDescriptor("aircrack-ng", ToolCategory.WIRELESS,
                   "Complete suite of tools for 802.11 wireless network security",
                   "aircrack-ng", ["kali", "blackarch"], "https://www.aircrack-ng.org"),
    ToolDescriptor("kismet", ToolCategory.WIRELESS,
                   "Wireless network and device detector/sniffer",
                   "kismet", ["kali", "blackarch"], "https://www.kismetwireless.net"),

    # Reverse Engineering
    ToolDescriptor("ghidra", ToolCategory.REVERSE_ENGINEERING,
                   "NSA software reverse engineering suite",
                   "ghidra", ["kali", "blackarch"], "https://ghidra-sre.org"),
    ToolDescriptor("gdb", ToolCategory.REVERSE_ENGINEERING,
                   "GNU debugger",
                   "gdb", ["kali", "blackarch"], "https://www.gnu.org/software/gdb"),
    ToolDescriptor("radare2", ToolCategory.REVERSE_ENGINEERING,
                   "Reverse engineering framework and CLI tools",
                   "radare2", ["kali", "blackarch"], "https://rada.re"),
    ToolDescriptor("binwalk", ToolCategory.REVERSE_ENGINEERING,
                   "Firmware analysis tool",
                   "binwalk", ["kali", "blackarch"], "https://github.com/ReFirmLabs/binwalk"),

    # Exploitation
    ToolDescriptor("msfconsole", ToolCategory.EXPLOITATION,
                   "Metasploit Framework console",
                   "msfconsole", ["kali", "blackarch"], "https://www.metasploit.com"),
    ToolDescriptor("pwncat", ToolCategory.EXPLOITATION,
                   "Enhanced netcat for post-exploitation",
                   "pwncat", ["blackarch"], "https://github.com/calebstewart/pwncat"),

    # Sniffing & Spoofing
    ToolDescriptor("wireshark", ToolCategory.SNIFFING_SPOOFING,
                   "Network protocol analyser",
                   "wireshark", ["kali", "blackarch"], "https://www.wireshark.org"),
    ToolDescriptor("tcpdump", ToolCategory.SNIFFING_SPOOFING,
                   "Command-line packet analyser",
                   "tcpdump", ["kali", "blackarch"], "https://www.tcpdump.org"),
    ToolDescriptor("ettercap", ToolCategory.SNIFFING_SPOOFING,
                   "Man-in-the-middle attacks on LAN",
                   "ettercap", ["kali", "blackarch"], "https://www.ettercap-project.org"),

    # Forensics
    ToolDescriptor("volatility3", ToolCategory.FORENSICS,
                   "Advanced memory forensics framework",
                   "volatility3", ["kali", "blackarch"], "https://github.com/volatilityfoundation/volatility3"),
    ToolDescriptor("autopsy", ToolCategory.FORENSICS,
                   "Digital forensics platform and graphical interface",
                   "autopsy", ["kali"], "https://www.autopsy.com"),
    ToolDescriptor("foremost", ToolCategory.FORENSICS,
                   "File carver for recovering files",
                   "foremost", ["kali", "blackarch"], ""),

    # Social Engineering
    ToolDescriptor("setoolkit", ToolCategory.SOCIAL_ENGINEERING,
                   "Social-Engineer Toolkit",
                   "set", ["kali", "blackarch"], "https://github.com/trustedsec/social-engineer-toolkit"),
]


class LinuxToolkit:
    """
    Catalogue and runner for Kali-Linux and Black Arch security tools.

    Usage::

        toolkit = LinuxToolkit()
        tools = toolkit.list_tools(category=ToolCategory.WEB_APPLICATION)
        for t in tools:
            print(t.name, t.is_installed())
    """

    def __init__(self) -> None:
        self._catalogue = {t.name: t for t in _CATALOGUE}

    # ------------------------------------------------------------------
    # Catalogue queries
    # ------------------------------------------------------------------

    def list_tools(
        self,
        category: ToolCategory | None = None,
        distro: str | None = None,
        installed_only: bool = False,
    ) -> list[ToolDescriptor]:
        """
        List tools, optionally filtered by category, distro, or install status.
        """
        tools = list(self._catalogue.values())
        if category is not None:
            tools = [t for t in tools if t.category == category]
        if distro is not None:
            tools = [t for t in tools if distro.lower() in t.distros]
        if installed_only:
            tools = [t for t in tools if t.is_installed()]
        return tools

    def get_tool(self, name: str) -> ToolDescriptor | None:
        return self._catalogue.get(name)

    def categories(self) -> list[ToolCategory]:
        return list(ToolCategory)

    def summary(self) -> dict[str, Any]:
        all_tools = list(self._catalogue.values())
        return {
            "total": len(all_tools),
            "installed": sum(1 for t in all_tools if t.is_installed()),
            "kali_tools": sum(1 for t in all_tools if "kali" in t.distros),
            "blackarch_tools": sum(1 for t in all_tools if "blackarch" in t.distros),
        }
