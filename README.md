# SchedCP - Automatically Optimize Linux Scheduler with MCP Server

> WIP: We are building a benchmark for evaluating the optimizations for OS!

SchedCP is an `experimental` project that enables AI optimization of Linux kernel schedulers using the sched-ext framework. It provides e2e automatic scheduler selection/synthesis, workload profiling, and performance optimization without any human intervention or guidance.

Paper: [SchedCP: Towards Agentic OS](https://arxiv.org/abs/2509.01245)

*The future is not just about letting AI write code for you; the AI agent should act as your system administrator, able to optimize anything for you automatically, without requiring any manual intervention!*

It includes the following tools:

- **autotune** - AI Agent-powered automatic OS optimization
- **schedcp** - MCP server for scheduler management and generation

## Demo

Start optimize any workload with AI by simply run:

```sh
autotune/target/release/autotune cc "<your workload command>"
# example for linux build
autotune/target/release/autotune cc "make -C workloads/linux-build-bench/linux clean -j && make -C workloads/linux-build-bench/linux -j" 
# example for schbench
autotune/target/release/autotune cc  workloads/basic/schbench/schbench
```

Allow LLM Agent to auto select and config the best scheduler:

![document/schbench-optimize.gif](https://github.com/eunomia-bpf/schedcp/blob/master/document/schbench-optimize.gif?raw=true)

Allow LLM Agents to write new schedulers:

## Features & design

- Automatic workload profiling
- Automatic scheduler selection based on workload characteristics
- Performance tracking across different schedulers
- Real-time scheduler management and generation

![document/design.png](https://github.com/eunomia-bpf/schedcp/blob/master/document/design.png?raw=true)

The current MCP tools include:

- **list_schedulers** - Get detailed information about all available schedulers
- **run_scheduler** - Start schedulers with custom configurations
- **stop_scheduler** - Stop running scheduler instances
- **get_execution_status** - Monitor scheduler performance and output
- **create_and_verify_scheduler** - Create custom BPF schedulers from source code
- **system_monitor** - Collect real-time CPU, memory, and scheduler metrics
- **workload** - Manage workload profiles and execution history

## Installation

### Requirements

- Linux kernel 6.12+ with sched-ext support  
- Rust toolchain

The major dependencies are the dependencies for the sched-ext framework. You can check the [github.com/sched-ext/scx](https://github.com/sched-ext/scx) for more details.

You also need to install the deps for the workloads you want to optimize.

### Build

```bash
# Clone with submodules
git clone https://github.com/eunomia-bpf/schedcp
cd schedcp
git submodule update --init --recursive scheduler/scx

# Build schedulers
cd scheduler && make && make install && cd ..
# Build autotune
cd autotune && cargo build --release && cd ..
# Build MCP server
cd mcp && cargo build --release && cd ..
```

## Documentation

### User Guides

- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Complete guide on how to use schedulers
  - Installation and setup
  - Using schedulers with CLI, MCP server, and autotune
  - Understanding scheduler selection
  - Troubleshooting

- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Detailed project organization
  - Component overview and responsibilities
  - Directory structure and file locations
  - Data flow and integration points
  - Build system architecture

- **[AI_AGENTS.md](AI_AGENTS.md)** - AI agent implementation
  - How observation, planning, execution, and learning agents work
  - MCP tools and agent capabilities
  - Example optimization workflows
  - Integration with Claude and other AI systems

### Design Documents

- **[document/sched-agent-design.md](document/sched-agent-design.md)** - Multi-agent framework design
- **[document/schedcp-design.md](document/schedcp-design.md)** - System control plane architecture
- **Research Paper**: [SchedCP: Towards Agentic OS](https://arxiv.org/abs/2509.01245)

## Quick Start

**You should run the claude-code on project root directory.**

### Autotune (Recommended)

```bash
# Set sudo password
export SCHEDCP_SUDO_PASSWORD="your_password"

# Optimize any workload
./autotune/target/release/autotune cc "<your workload command>"
```

### MCP Server

check the [.mcp.json](https://github.com/eunomia-bpf/schedcp/blob/master/.mcp.json) for more details. You can just open the claude-code on the 

### CLI Tool

```bash
export SCHEDCP_SUDO_PASSWORD="your_password"

# List schedulers
./mcp/target/release/schedcp-cli list

# Run a scheduler
./mcp/target/release/schedcp-cli run scx_rusty --sudo

# Monitor system metrics
./mcp/target/release/schedcp-cli monitor
```

For detailed usage instructions, see [USAGE_GUIDE.md](USAGE_GUIDE.md).

## Artifact from Paper

Artifact for reproducing results from **"Towards Agentic OS: An LLM Agent Framework for Linux Schedulers"** (arXiv:2509.01245).

### Evaluation Workloads

#### 1. **Linux Kernel Build**
- **Location**: [`workloads/linux-build-bench/`](workloads/linux-build-bench/)
- **Run**: `autotune/target/release/autotune cc "make -C workloads/linux-build-bench/linux -j"`

#### 2. **schbench**
- **Location**: [`workloads/basic/schbench_test/`](workloads/basic/schbench_test/)
- **Run**: `autotune/target/release/autotune cc "workloads/basic/schbench/schbench"`

#### 3. **Batch Workloads**
- **Location**: [`workloads/processing/`](workloads/processing/)
- **Run**: See [`workloads/processing/README.md`](workloads/processing/README.md). The schedulers there are already generated by Claude Code. The original version was letting Claude Code generate them first and run the workloads with benchmarks mannually.
    - Workloads: [workloads/processing/assets](workloads/processing/assets/)
    - Prompt and testcases: [workloads/processing/scripts](workloads/processing/scripts)
    - Generated schedulers: [workloads/processing/schedulers](workloads/processing/schedulers)

## Related Projects

- [sched-ext](https://github.com/sched-ext/scx) - Linux kernel BPF scheduler framework
- [Model Context Protocol](https://modelcontextprotocol.io/) - AI-application integration protocol

## License

See [LICENSE](https://github.com/eunomia-bpf/schedcp/blob/master/LICENSE) for details.
