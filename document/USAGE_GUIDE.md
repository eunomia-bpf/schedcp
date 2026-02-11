# SchedCP Usage Guide

This guide explains how to use SchedCP to optimize Linux kernel schedulers for your workloads.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Using Schedulers](#using-schedulers)
- [Advanced Usage](#advanced-usage)
- [Understanding Scheduler Selection](#understanding-scheduler-selection)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Before using SchedCP, ensure your system meets the following requirements:

### System Requirements

- **Linux kernel 6.12+** with sched-ext support enabled
- **Root/sudo access** for loading kernel schedulers
- **CPU architecture**: x86_64 (tested) or aarch64

### Checking sched-ext Support

To verify your kernel has sched-ext support:

```bash
# Check if sched_ext is available
grep -r "sched_ext" /boot/config-$(uname -r) || \
  grep -r "CONFIG_SCHED_CLASS_EXT" /boot/config-$(uname -r)

# Check if BPF is enabled
grep -r "CONFIG_BPF" /boot/config-$(uname -r)
```

### Required Software

- **Rust toolchain** (1.82+): `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **Clang/LLVM** (16+, 17 recommended): `sudo apt install clang llvm libbpf-dev` (Ubuntu/Debian)
- **Meson build system** (1.2.0+): `pip install meson`
- **libbpf** (1.2.2+): Usually available via package manager

## Installation

### Quick Installation

```bash
# Clone the repository with submodules
git clone https://github.com/eunomia-bpf/schedcp
cd schedcp
git submodule update --init --recursive scheduler/scx

# Build schedulers (this will take several minutes)
cd scheduler
make deps    # Install dependencies
make         # Build all schedulers
make install # Install to ~/.schedcp/scxbin/
cd ..

# Build the MCP server and CLI tool
cd mcp
cargo build --release
cd ..

# Build the autotune tool (optional but recommended)
cd autotune
cargo build --release
cd ..
```

After installation, the following binaries will be available:
- `mcp/target/release/schedcp` - MCP server for AI integration
- `mcp/target/release/schedcp-cli` - Command-line scheduler management tool
- `autotune/target/release/autotune` - AI-powered workload optimizer
- `~/.schedcp/scxbin/*` - All compiled scheduler binaries

### Setting up Sudo Access

SchedCP needs sudo privileges to load kernel schedulers. You have two options:

#### Option 1: Environment Variable (for development/testing)

```bash
export SCHEDCP_SUDO_PASSWORD="your_password"
```

Add this to your `~/.bashrc` or `~/.zshrc` for persistence.

#### Option 2: Passwordless Sudo (recommended for production)

```bash
# Edit sudoers file
sudo visudo

# Add this line (replace 'username' with your username):
username ALL=(ALL) NOPASSWD: /home/username/.schedcp/scxbin/*
```

## Quick Start

The fastest way to optimize a workload is using the autotune tool:

```bash
# Set sudo password if not using passwordless sudo
export SCHEDCP_SUDO_PASSWORD="your_password"

# Optimize any command
./autotune/target/release/autotune cc "make -j$(nproc)"

# Or for a specific workload
./autotune/target/release/autotune cc "workloads/basic/schbench/schbench"
```

The autotune tool will:
1. Analyze your workload characteristics
2. Select and test multiple schedulers
3. Recommend the optimal scheduler configuration
4. Apply the best scheduler automatically

## Using Schedulers

### Method 1: Using schedcp-cli (Recommended for Direct Control)

The CLI tool provides direct control over scheduler management:

#### List Available Schedulers

```bash
# List all schedulers with detailed information
./mcp/target/release/schedcp-cli list

# Filter by name (supports partial matching)
./mcp/target/release/schedcp-cli list --name rusty

# List only production-ready schedulers
./mcp/target/release/schedcp-cli list --production

# Combine filters
./mcp/target/release/schedcp-cli list --production --name simple
```

#### Run a Scheduler

```bash
# Run with default parameters
./mcp/target/release/schedcp-cli run scx_rusty --sudo

# Run with custom parameters
./mcp/target/release/schedcp-cli run scx_rusty --sudo -- --slice-us 5000

# Run for a specific duration then exit
timeout 60 ./mcp/target/release/schedcp-cli run scx_bpfland --sudo
```

#### Monitor System Metrics

```bash
# Monitor for default 10 seconds
./mcp/target/release/schedcp-cli monitor

# Monitor for custom duration
./mcp/target/release/schedcp-cli monitor --duration 30

# Example output:
# CPU Utilization:
#   Average: 45.2%
#   Maximum: 78.5%
# Memory Usage:
#   Average: 6234 MB (38.5%)
#   Maximum: 7123 MB (44.0%)
# Scheduler Statistics:
#   Total timeslices: 45231
#   Total run time: 42.3s
```

### Method 2: Using the MCP Server with AI Assistants

The MCP server enables AI assistants (like Claude) to manage schedulers intelligently:

#### Starting the MCP Server

```bash
# Set sudo password
export SCHEDCP_SUDO_PASSWORD="your_password"

# Start the MCP server
./mcp/target/release/schedcp
```

#### Configuring Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "schedcp": {
      "command": "/absolute/path/to/schedcp/mcp/target/release/schedcp",
      "env": {
        "SCHEDCP_SUDO_PASSWORD": "your_password"
      }
    }
  }
}
```

#### Using with Claude Code

If you're using Claude Code (VSCode extension), the `.mcp.json` file in the project root is already configured. Just open the project in Claude Code.

#### Example AI Interactions

Once configured, you can ask Claude:

- "List all production-ready schedulers suitable for gaming workloads"
- "Run the scx_bpfland scheduler with parameters optimized for low latency"
- "Create a workload profile for my web server and test different schedulers"
- "Monitor system performance while running my benchmark"
- "What scheduler should I use for a machine learning training workload?"

### Method 3: Using Autotune for Automated Optimization

The autotune tool combines workload analysis with scheduler optimization:

#### Basic Command Optimization

```bash
# Analyze and optimize any command
./autotune/target/release/autotune run "make -j8"

# Interactive optimization session
./autotune/target/release/autotune run --interactive "python train.py"
```

#### Scheduler Optimization (cc command)

```bash
# Automatic scheduler optimization
./autotune/target/release/autotune cc "<your workload command>"

# Examples:
./autotune/target/release/autotune cc "make -C workloads/linux-build-bench/linux -j"
./autotune/target/release/autotune cc "workloads/basic/schbench/schbench"
./autotune/target/release/autotune cc "python workloads/pytorch/train.py"
```

The `cc` command performs complete scheduler optimization:
1. **Workload Profiling**: Analyzes the command to understand its characteristics
2. **Scheduler Selection**: Identifies candidate schedulers based on workload type
3. **Automated Testing**: Tests multiple schedulers with optimal configurations
4. **Performance Comparison**: Measures and compares execution time and system metrics
5. **Recommendation**: Suggests the best scheduler with specific parameters

## Advanced Usage

### Creating Custom Schedulers

You can create custom BPF schedulers from source code:

```bash
# Using CLI
./mcp/target/release/schedcp-cli create-and-run /path/to/my_scheduler.bpf.c

# The scheduler will be:
# 1. Compiled with clang
# 2. Verified by loading in kernel for 10 seconds
# 3. Available for use like built-in schedulers
```

Custom schedulers must include these BPF operations:
- `select_cpu` - Select CPU for task placement
- `enqueue` - Enqueue task for scheduling
- `dispatch` - Dispatch task to run
- `init` - Initialize scheduler
- `exit` - Clean up scheduler

See `scheduler/scx/scheds/c/` for examples.

### Working with Workload Profiles

Workload profiles help track scheduler performance across different workloads:

#### Using the MCP Server

The AI assistant can:
- Create workload profiles with natural language descriptions
- Track execution history for each profile
- Make recommendations based on historical performance

Example workflow:
1. Create profile: "Web server handling 10K concurrent connections"
2. Test schedulers: Run multiple schedulers and record results
3. View history: See which schedulers performed best
4. Get recommendations: AI suggests optimal scheduler based on history

#### Storage

Workload profiles are stored in `schedcp_workloads.json` in the working directory.

### Understanding Scheduler Parameters

Each scheduler has different tuning parameters. Use the list command to see details:

```bash
./mcp/target/release/schedcp-cli list --name scx_rusty
```

Common parameter types:

- **Time slices** (slice_us): How long tasks run before preemption
  - Lower values: Better responsiveness, higher overhead
  - Higher values: Better throughput, lower responsiveness

- **Domain configuration**: How CPUs are grouped
  - Options: auto, performance, powersave, all, none

- **Preemption control**: Whether tasks can be interrupted
  - Enabled: Better latency, more context switches
  - Disabled: Better throughput, potential latency spikes

- **CPU frequency control**: Dynamic frequency scaling
  - Enabled: Power savings, potential performance impact
  - Disabled: Consistent performance, higher power usage

### Production-Ready Schedulers

These schedulers are stable and recommended for production use:

- **scx_rusty**: Multi-domain scheduler with intelligent load balancing
  - Best for: General-purpose workloads, multi-socket systems
  - Characteristics: Domain-aware, load balancing, NUMA optimization

- **scx_bpfland**: Interactive workload prioritization
  - Best for: Gaming, live streaming, multimedia, real-time audio
  - Characteristics: Voluntary context switch detection, interactive task priority

- **scx_lavd**: Latency-aware scheduler
  - Best for: Low-latency workloads, gaming, real-time applications
  - Characteristics: Latency-aware virtual deadline scheduling

- **scx_simple**: Simple scheduler for single-socket systems
  - Best for: Simple workloads, single-socket systems, debugging
  - Characteristics: FIFO-based, minimal overhead

- **scx_layered**: Multi-layer scheduler with high configurability
  - Best for: Complex workloads with multiple priority levels
  - Characteristics: Layer-based priorities, highly configurable

- **scx_flatcg**: High-performance cgroup-aware scheduler
  - Best for: Containerized workloads, cgroup hierarchies
  - Characteristics: Cgroup-aware, container optimization

- **scx_nest**: Frequency-optimized scheduler
  - Best for: Low CPU utilization, power-sensitive workloads
  - Characteristics: Core nesting, frequency optimization

- **scx_flash**: EDF (Earliest Deadline First) scheduler
  - Best for: Predictable latency requirements
  - Characteristics: Deadline-based, latency predictability

- **scx_p2dq**: Versatile scheduler with pick-two load balancing
  - Best for: Diverse workloads, general-purpose use
  - Characteristics: Pick-two load balancing, versatile

## Understanding Scheduler Selection

### Workload Characteristics

Different workloads benefit from different schedulers:

#### Latency-Sensitive Workloads
- **Examples**: Gaming, audio processing, real-time systems
- **Recommended**: scx_lavd, scx_bpfland, scx_flash
- **Why**: These prioritize quick response times and minimize scheduling delay

#### Throughput-Focused Workloads
- **Examples**: Batch processing, compilation, scientific computing
- **Recommended**: scx_rusty, scx_p2dq
- **Why**: These optimize for overall work completion and CPU utilization

#### Interactive Workloads
- **Examples**: Desktop applications, web browsers, IDEs
- **Recommended**: scx_bpfland, scx_lavd
- **Why**: These detect and prioritize interactive tasks

#### Containerized Workloads
- **Examples**: Docker, Kubernetes, microservices
- **Recommended**: scx_flatcg
- **Why**: Optimized for cgroup hierarchies and container isolation

#### Mixed Workloads
- **Examples**: Development machines, multi-user systems
- **Recommended**: scx_layered, scx_p2dq
- **Why**: These can handle diverse task types with configurable policies

### How the AI Selects Schedulers

The autotune tool and MCP server use this process:

1. **Workload Analysis**: Examines command, file access patterns, resource usage
2. **Classification**: Categorizes as latency-sensitive, throughput-focused, etc.
3. **Candidate Selection**: Identifies 3-5 schedulers from metadata in `scheduler/schedulers.json`
4. **Configuration**: Applies workload-specific tuning parameters
5. **Testing**: Runs workload with each scheduler and measures performance
6. **Recommendation**: Selects best based on metrics (latency, throughput, CPU usage)

### Scheduler Metadata

Scheduler characteristics are defined in `scheduler/schedulers.json`:

```json
{
  "name": "scx_bpfland",
  "production_ready": true,
  "description": "...",
  "use_cases": ["gaming", "live_streaming", "multimedia"],
  "algorithm": "vruntime_based",
  "characteristics": "interactive task prioritization, ...",
  "tuning_parameters": { ... }
}
```

This metadata enables intelligent scheduler selection by AI assistants.

## Troubleshooting

### Common Issues

#### "Permission denied" when loading scheduler

**Problem**: Scheduler fails to load due to insufficient permissions.

**Solution**:
```bash
# Verify sudo access
sudo -v

# Set password environment variable
export SCHEDCP_SUDO_PASSWORD="your_password"

# Or configure passwordless sudo (see Installation section)
```

#### "sched_ext not supported" error

**Problem**: Kernel doesn't have sched-ext support.

**Solution**:
- Upgrade to Linux kernel 6.12 or newer
- Ensure kernel is compiled with CONFIG_SCHED_CLASS_EXT=y
- Check with: `grep SCHED_CLASS_EXT /boot/config-$(uname -r)`

#### Scheduler compilation fails

**Problem**: Building schedulers fails with compilation errors.

**Solution**:
```bash
# Install dependencies
cd scheduler
make deps

# Check clang version (need 16+)
clang --version

# Update if necessary
sudo apt install clang-17 llvm-17 libbpf-dev

# Clean and rebuild
make clean
make
```

#### MCP server not connecting

**Problem**: Claude Desktop can't connect to MCP server.

**Solution**:
- Verify the binary path in `claude_desktop_config.json` is absolute
- Check environment variables are set correctly
- Look at MCP server logs in `schedcp.log`
- Restart Claude Desktop after configuration changes

#### Scheduler causes system instability

**Problem**: System becomes unresponsive after loading scheduler.

**Solution**:
- Press Ctrl+C to stop the scheduler immediately
- The default scheduler will be restored automatically
- Report the issue with scheduler name and workload details
- Use production-ready schedulers for critical systems

### Getting Help

- **Documentation**: See other `.md` files in the repository
- **Issues**: https://github.com/eunomia-bpf/schedcp/issues
- **sched-ext upstream**: https://github.com/sched-ext/scx
- **Paper**: https://arxiv.org/abs/2509.01245

### Debug Logging

Enable detailed logging:

```bash
# For MCP server
RUST_LOG=debug ./mcp/target/release/schedcp

# For CLI tool
RUST_LOG=debug ./mcp/target/release/schedcp-cli list

# Check logs
tail -f schedcp.log
```

### Performance Monitoring

Monitor scheduler performance in real-time:

```bash
# Using scxtop (if built)
sudo ./scheduler/tools/scxtop

# Using system monitor
./mcp/target/release/schedcp-cli monitor --duration 60

# Using standard tools
perf stat -a sleep 10
```

## Next Steps

- Read [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) to understand the codebase
- Read [AI_AGENTS.md](AI_AGENTS.md) to learn about AI integration
- Explore example workloads in `workloads/` directory
- Read the research paper for theoretical background
- Join the community and contribute!
