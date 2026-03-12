# Cadis

## Prerequisites

Before installing, ensure your systems meets the following requirements:
- **OS:** Linux or WSL
- **Python:** Version 3.10 or higher
- **CUDA:** 12.2
- **NVIDIA Driver:** 535 or higher
- **Overlay Network Cluster** A populated cluster of devices using either Tailscale or Netbird

## Setup

Currently, Cadis relies on specific changes to the underlying inference engine. You will need to clone this repository alongside a forked repository of your desired inference engine.

*(Note: We currently have pending pull requests for `sglang` and `vLLM`. If they are not accepted, we will introduce these changes as monkey patches directly within Cadis in the future.)*

### 1. Clone Repositories

Depending on your engine of choice, clone the relevant fork so it sits in the same parent directory as `cadis`:

**For vLLM:**
```bash
git clone https://github.com/JorgenTrondsen/vllm.git
```

**For SGLang:**
```bash
git clone -b pp_optimization https://github.com/JorgenTrondsen/sglang.git
```

Your folder structure must look like this:
```text
person@machine:~/JorgTrond$ ls
cadis  sglang  vllm
```

### 2. Install Build Tools

Upgrade `pip` and install `uv` (our recommended fast Python package installer):
```bash
pip install --upgrade pip
pip install uv
```

### 3. Install Your Preferred Inference Engine

Follow the steps below for the engine you intend to use.

#### Option A: SGLang Setup
```bash
cd sglang
uv venv venv
source venv/bin/activate
uv pip install -e "python"
```

#### Option B: vLLM Setup
```bash
cd vllm
uv venv venv
source venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install --prerelease=allow --editable .
```