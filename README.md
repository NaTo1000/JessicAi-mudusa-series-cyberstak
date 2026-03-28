# JessicAi-mudusa-series-cyberstak

A comprehensive cybersecurity and AI development suite integrating **AicodeX**, **PortmanAI**, full **Kali-Linux / Black Arch** toolkits, and a sandboxed Linux environment with on-demand VPS scaling.

---

## Features

### 🔧 AicodeX — Advanced Coding Assistant
- Multi-mode operation: **Coding**, **Prompted AI Design**, **Code Debug**
- Hotkey-activated sandbox with drag-and-drop AI model loading
- Smooth code generation and fine-tuning workflows

### 🛡️ PortmanAI — AI-Driven Debugging & Toolchain
- Intelligent debugger with root-cause analysis
- Seamless integration with the AicodeX coding environment
- AI-driven toolchain operations for security and DevOps

### 🐧 Linux Sandbox Environment
- Full **Kali-Linux** and **Black Arch** toolkit availability
- Sandboxed execution of pentesting, forensics, and exploitation tools
- Isolated container runtime — tools run without polluting the host

### 🤖 Drag-and-Drop Model Runner
- Load any model from GitHub or HuggingFace by URL or repo path
- Automatic environment setup and sandboxed inference
- GPU/CPU detection with dynamic allocation

### ☁️ VPS Scaling
- On-demand CPU / GPU / RAM allocation
- Local ↔ VPS workload migration
- Real-time resource monitoring

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py --mode coding          # start in coding mode
python main.py --mode debug           # start in debug mode
python main.py --mode design          # start in prompted design mode
python main.py sandbox --model <url>  # launch sandbox with a model
```

---

## Project Layout

```
.
├── aicodex/           # AicodeX coding assistant & mode engine
├── portmanai/         # PortmanAI debugger & toolchain
├── sandbox/           # Sandboxed Linux + model-runner environment
│   └── linux_toolkit/ # Kali-Linux & Black Arch tool wrappers
├── vps/               # VPS scaling & resource management
├── config/            # YAML configuration files
├── tests/             # Unit tests
└── main.py            # CLI entry point
```

---

## Modes

| Mode | Hotkey | Description |
|------|--------|-------------|
| Coding | `Ctrl+1` | Full AicodeX code-generation and editing |
| Prompted Design | `Ctrl+2` | AI-driven architecture and design generation |
| Debug | `Ctrl+3` | PortmanAI-powered root-cause analysis and fixes |

---

## Security

- All sandbox operations run in isolated environments (Docker/nsjail)
- Model inference is sandboxed with strict filesystem and network limits
- Kali / Black Arch tools are invoked via a controlled API, never raw shell injection

---

## License

MIT