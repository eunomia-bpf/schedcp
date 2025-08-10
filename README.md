# SchedCP - MCP Server for Linux Scheduler Management

SchedCP is a Model Context Protocol (MCP) server that enables AI assistants to intelligently manage Linux kernel schedulers through the sched-ext framework. It provides AI-powered scheduler selection, workload profiling, and real-time performance optimization.

## Overview

SchedCP provides intelligent scheduler management through two main interfaces:
- **schedcp** - MCP server enabling AI assistants (like Claude) to manage schedulers programmatically
- **schedcp-cli** - Command-line tool for direct scheduler management and testing

The MCP server allows AI assistants to:
- Analyze workload characteristics and select optimal schedulers
- Create and manage workload profiles with natural language descriptions  
- Track scheduler performance across different workloads
- Make data-driven scheduler recommendations based on execution history

## Features

- **AI-Powered Scheduler Selection**: Natural language workload descriptions for intelligent scheduler recommendations
- **Embedded Schedulers**: All sched-ext schedulers and configuration embedded in binaries
- **Workload Profiling**: Create and manage workload profiles with execution history tracking
- **Real-time Management**: Start, stop, and monitor sched-ext schedulers dynamically
- **Performance Tracking**: Capture and analyze scheduler performance metrics across different workloads
- **Smart Filtering**: Filter schedulers by name or production readiness
- **Production Ready**: Clear indication of which schedulers are ready for production use
- **MCP Integration**: Seamless integration with AI assistants like Claude for automated optimization

## Quick Start

### Prerequisites

- Linux kernel 6.12+ with sched-ext support
- Clang/LLVM >= 16 (17 recommended)  
- Rust toolchain >= 1.82
- libbpf >= 1.2.2

### Building schedcp

```bash
# Clone the repository
git clone https://github.com/eunomia-bpf/schedcp
cd schedcp

# Build the MCP server and CLI
cd mcp
cargo build --release
# The binaries will be at:
# - mcp/target/release/schedcp (MCP server)
# - mcp/target/release/schedcp-cli (CLI tool)
```

### Using the MCP Server with AI Assistants

#### Installation for Claude Desktop

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "schedcp": {
      "command": "/path/to/schedcp/mcp/target/release/schedcp",
      "env": {
        "SCHEDCP_SUDO_PASSWORD": "your_password"
      }
    }
  }
}
```

#### Available MCP Tools

The MCP server provides these tools for AI assistants:

**Scheduler Management:**
- `list_schedulers` - List schedulers with detailed information
- `run_scheduler` - Start a scheduler with specified parameters
- `stop_scheduler` - Stop a running scheduler
- `get_execution_status` - Get status of a scheduler execution

**Workload Profile Management:**
- `create_workload_profile` - Create workload profiles from natural language descriptions
- `add_execution_history` - Record performance results for workloads
- `list_workload_profiles` - List all workload profiles
- `get_workload_history` - View execution history for workloads

#### AI-Assisted Workflow Example

1. **Describe your workload**: "I'm running a web server with high concurrent connections"
2. **AI analyzes and recommends**: AI creates workload profile and suggests `scx_bpfland` for interactive workloads
3. **Test and track**: AI runs scheduler, monitors performance, and records results
4. **Optimize iteratively**: AI compares results and fine-tunes scheduler selection

### Using schedcp CLI

```bash
# Set sudo password for scheduler management
export SCHEDCP_SUDO_PASSWORD="your_password"

# List all schedulers with full details
./mcp/target/release/schedcp-cli list

# Filter schedulers by name or production readiness
./mcp/target/release/schedcp-cli list --name rusty --production

# Run a scheduler with default parameters
./mcp/target/release/schedcp-cli run scx_rusty --sudo

# Run with custom scheduler parameters  
./mcp/target/release/schedcp-cli run scx_rusty --sudo -- --slice-us 20000 --fifo-sched

# Check scheduler execution status
./mcp/target/release/schedcp-cli status
```

### Building Kernel Schedulers (Optional)

To build the actual kernel schedulers that schedcp manages:

```bash
# From the project root, build all schedulers
cd scheduler
make deps      # Install dependencies (first time only)
make build     # Build all schedulers (outputs to sche_bin/)

# Or build specific types
make build-c    # Build only C schedulers
make build-rust # Build only Rust schedulers
make build-tools # Build tools (scxtop, scxctl, scx_loader)
```

**Note**: The MCP server includes embedded scheduler binaries, so building from source is optional unless you want to modify schedulers.

## Available Schedulers

SchedCP includes comprehensive information about sched-ext schedulers:

### Production-Ready Schedulers
- **scx_rusty** - Multi-domain scheduler with intelligent load balancing
- **scx_simple** - Simple scheduler for single-socket systems  
- **scx_lavd** - Latency-aware scheduler for gaming and interactive workloads
- **scx_bpfland** - Interactive workload prioritization
- **scx_layered** - Highly configurable multi-layer scheduler
- **scx_flatcg** - High-performance cgroup-aware scheduler
- **scx_nest** - Frequency-optimized scheduler for low CPU utilization
- **scx_flash** - EDF scheduler for predictable latency

### Experimental Schedulers
- Various experimental and educational schedulers for testing and development

Use `./mcp/target/release/schedcp-cli list --production` to see only production-ready schedulers.

## Architecture

schedcp combines multiple technologies to create an intelligent kernel scheduler management system:

- **MCP (Model Context Protocol)**: Enables AI assistants to interact with schedulers programmatically
- **eBPF/sched_ext**: Runtime kernel programmability for schedulers and monitoring
- **Embedded Resources**: All scheduler binaries and metadata embedded using rust-embed
- **Workload Profiling**: Natural language descriptions mapped to scheduler performance history
- **Real-time Management**: Async Rust runtime for concurrent scheduler operations

## Key Features

- **AI-Driven Optimization**: Natural language workload descriptions enable intelligent scheduler selection
- **Persistent Workload Profiles**: Track performance history across different workloads and schedulers  
- **Real-time Scheduler Management**: Start, stop, and monitor schedulers with comprehensive status reporting
- **Production-Ready Filtering**: Clear indicators of which schedulers are suitable for production environments
- **Embedded Scheduler Database**: Complete scheduler information and binaries included in the MCP server
- **Comprehensive Logging**: Detailed execution logs for debugging and performance analysis
- **Safety Mechanisms**: Controlled scheduler execution with proper cleanup and error handling

## Documentation

- [MCP Server Details](mcp/README.md) - Detailed MCP server documentation and examples
- [Scheduler Guide](CLAUDE.md) - Working with sched-ext schedulers  
- [Developer Guide](scheduler/scx/DEVELOPER_GUIDE.md) - Contributing to the sched-ext framework
- [Workload Examples](workloads/) - Sample workloads and benchmarking tools

## Related Projects

- [sched-ext](https://github.com/sched-ext/scx) - Linux kernel scheduler framework
- [Model Context Protocol](https://modelcontextprotocol.io/) - Protocol for AI-application integration

## Requirements

- Linux kernel 6.12+ with sched_ext support
- Clang/LLVM >= 16 (17 recommended)
- Rust toolchain >= 1.82
- libbpf >= 1.2.2

## License

schedcp is open source software. See LICENSE for details.