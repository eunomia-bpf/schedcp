# SchedCP Project Structure

This document provides a detailed overview of the SchedCP repository structure, explaining what each component does and where to find specific functionality.

## Table of Contents

- [Overview](#overview)
- [Root Directory](#root-directory)
- [MCP Server (`mcp/`)](#mcp-server-mcp)
- [Scheduler System (`scheduler/`)](#scheduler-system-scheduler)
- [Autotune Tool (`autotune/`)](#autotune-tool-autotune)
- [Workloads (`workloads/`)](#workloads-workloads)
- [Documentation (`document/`)](#documentation-document)
- [Data Flow](#data-flow)
- [Build System](#build-system)

## Overview

SchedCP is organized into several major components:

```
schedcp/
├── mcp/                    # MCP server for AI-assisted optimization
├── scheduler/              # Scheduler build system and metadata
├── autotune/               # AI-powered workload optimizer
├── workloads/              # Benchmark workloads for testing
├── document/               # Research papers and design docs
└── Configuration files     # Build and project configuration
```

Each component is relatively independent but they work together to provide end-to-end scheduler optimization.

## Root Directory

### Configuration Files

- **`.mcp.json`**: MCP server configuration for Claude Code integration
  - Specifies the schedcp MCP server command and arguments
  - Used by Claude Code to connect to the MCP server
  - Location: Project root

- **`.gitmodules`**: Git submodule configuration
  - Points to `scheduler/scx` - the sched-ext framework
  - Submodule URL: https://github.com/sched-ext/scx

- **`Makefile`**: Top-level build orchestration
  - Simple wrapper that delegates to component Makefiles
  - Usage: `make` (builds all components)

- **`.gitignore`**: Specifies files to ignore in git
  - Build artifacts, target directories, logs
  - Temporary files and compiled binaries

### Documentation Files

- **`README.md`**: Main project introduction and quick start
- **`document/USAGE_GUIDE.md`**: Comprehensive guide on using schedulers
- **`document/PROJECT_STRUCTURE.md`**: This file - codebase organization
- **`document/AI_AGENTS.md`**: AI agent implementation and architecture
- **`CLAUDE.md`**: Instructions for Claude AI when working with code
- **`LICENSE`**: Project license information

## MCP Server (`mcp/`)

The MCP (Model Context Protocol) server enables AI assistants to manage schedulers programmatically. This is the core integration point for AI-powered optimization.

### Directory Structure

```
mcp/
├── src/                        # Source code
│   ├── main.rs                 # MCP server entry point
│   ├── lib.rs                  # Core MCP protocol implementation
│   ├── cli.rs                  # CLI tool implementation
│   ├── scheduler_manager.rs   # Scheduler lifecycle management
│   ├── scheduler_generator.rs # Custom scheduler creation
│   ├── system_monitor.rs      # Real-time metrics collection
│   ├── workload_profile.rs    # Workload classification
│   └── storage.rs             # Persistent data storage
├── lib/                        # Supporting libraries
│   └── process_manager/       # Process management utilities
├── new_sched/                  # Custom scheduler working directory
│   ├── loader                 # BPF loader for custom schedulers
│   └── *.bpf.{c,o}           # Custom scheduler sources/objects
├── tests/                      # Integration tests
├── Cargo.toml                 # Rust dependencies and build config
└── README.md                  # MCP server documentation
```

### Key Components

#### `main.rs` - MCP Server Entry Point
- Starts the MCP server process
- Handles stdio-based Model Context Protocol communication
- Registers all MCP tools (list_schedulers, run_scheduler, etc.)
- Sets up logging to `schedcp.log`

#### `cli.rs` - CLI Tool Implementation
- Command-line interface for direct scheduler management
- Commands: `list`, `run`, `create-and-run`, `monitor`
- Uses the same core libraries as the MCP server
- Useful for testing and scripting without AI assistance

#### `scheduler_manager.rs` - Scheduler Lifecycle
- **Core responsibility**: Manage running schedulers
- **Key functions**:
  - `list_schedulers()`: Query scheduler metadata from embedded resources
  - `create_execution()`: Start a scheduler with specified parameters
  - `stop_execution()`: Gracefully stop a running scheduler
  - `get_execution_status()`: Check scheduler status and output
  - `create_and_verify_scheduler()`: Create custom schedulers
- **Embedded resources**: All scheduler binaries and metadata are embedded using `rust-embed`
- **Process management**: Uses `lib/process_manager` for subprocess control

#### `scheduler_generator.rs` - Custom Scheduler Creation (Private Module)
- **Core responsibility**: Create and compile custom BPF schedulers
- **Key functions** (accessed through scheduler_manager):
  - `compile_scheduler()`: Compile `.bpf.c` to `.bpf.o` using clang
  - `verify_scheduler()`: Load scheduler in kernel for 10 seconds to test
- **Build process**:
  - Uses clang with BPF target (`-target bpf`)
  - Includes scx headers from `scheduler/scx/scheds/include`
  - Generates BPF object files in `mcp/new_sched/`
  - Uses loader binary for kernel verification
- **Safety**: All custom schedulers must pass verification before use

#### `system_monitor.rs` - Metrics Collection
- **Core responsibility**: Collect real-time system metrics
- **Metrics collected**:
  - CPU utilization: user, system, idle, iowait (from `/proc/stat`)
  - Memory usage: total, free, available, buffers, cached (from `/proc/meminfo`)
  - Scheduler statistics: run time, wait time, timeslices (from `/proc/schedstat`)
- **Sampling**: Collects samples every 1 second asynchronously
- **Output**: Provides averages, maximums, and totals

#### `workload_profile.rs` - Workload Classification
- **Core responsibility**: Create and manage workload profiles
- **Profile structure**:
  - Unique identifier
  - Natural language description
  - Classification (latency-sensitive, throughput-focused, etc.)
  - Execution history (scheduler, parameters, results)
- **Storage**: Persisted in `schedcp_workloads.json`

#### `storage.rs` - Data Persistence
- **Core responsibility**: Save/load workload profiles
- **File format**: JSON (`schedcp_workloads.json`)
- **Location**: Current working directory
- **Auto-save**: Writes after every profile update

### Embedded Resources

The MCP server embeds all scheduler binaries and metadata:

- **Scheduler binaries**: Built from `scheduler/sche_bin/` and embedded at compile time
- **Metadata**: `scheduler/schedulers.json` embedded for offline access
- **Configuration**: Embedded default configurations

This makes the MCP server self-contained and deployable without local builds.

### MCP Tools

The server exposes these tools via Model Context Protocol:

1. **list_schedulers**: Query scheduler information
2. **run_scheduler**: Start scheduler with parameters
3. **stop_scheduler**: Stop running scheduler
4. **get_execution_status**: Check scheduler status
5. **create_and_verify_scheduler**: Create custom schedulers
6. **system_monitor**: Start/stop metrics collection
7. **workload**: Create/list/update workload profiles

See `src/lib.rs` for detailed tool implementations.

## Scheduler System (`scheduler/`)

The scheduler directory contains the build system for sched-ext schedulers and associated metadata.

### Directory Structure

```
scheduler/
├── scx/                        # sched-ext framework (git submodule)
│   ├── scheds/
│   │   ├── c/                 # C-based schedulers
│   │   ├── rust/              # Rust-based schedulers
│   │   └── include/           # BPF header files
│   ├── meson.build            # Build configuration
│   └── Documentation          # Upstream documentation
├── sche_bin/                   # Compiled scheduler binaries
├── sche_description/           # Scheduler documentation
│   ├── scx_rusty.md           # Individual scheduler docs
│   ├── scx_bpfland.md
│   └── ...
├── json/                       # JSON schema documentation
├── custom_schedulers/          # Custom scheduler examples
├── template/                   # Scheduler templates
├── ml-scheduler/               # ML-based scheduler research
├── scheduler_runner.py         # Python interface for schedulers
├── schedulers.json             # Scheduler metadata database
├── schedulers.json.backup_*    # Metadata backups
├── Makefile                    # Build automation
└── README.md                   # Build instructions
```

### Key Files

#### `schedulers.json` - Scheduler Metadata Database
The single source of truth for all scheduler information:

```json
{
  "schedulers": [
    {
      "name": "scx_bpfland",
      "production_ready": true,
      "description": "Interactive workload prioritization...",
      "use_cases": ["gaming", "multimedia"],
      "algorithm": "vruntime_based",
      "characteristics": "...",
      "tuning_parameters": {
        "slice_us": {
          "type": "integer",
          "description": "...",
          "default": 20000,
          "range": [1000, 100000]
        }
      }
    }
  ]
}
```

This metadata enables:
- AI-powered scheduler selection based on workload characteristics
- Automatic parameter tuning with valid ranges
- Production readiness filtering
- Natural language queries about schedulers

#### `Makefile` - Build Automation

Key targets:
- `make deps`: Install build dependencies
- `make`: Build all schedulers (C and Rust)
- `make build-c`: Build only C schedulers
- `make build-rust`: Build only Rust schedulers
- `make build-tools`: Build scx utilities (scx_loader, scxctl, scxtop)
- `make doc`: Generate scheduler documentation
- `make install`: Install schedulers to `~/.schedcp/scxbin/`
- `make clean`: Remove build artifacts
- `make update`: Update scx submodule

#### `scheduler_runner.py` - Python Interface

Provides a Python API for running schedulers:
- Used by workload benchmarking scripts
- Handles scheduler process management
- Captures output and metrics
- Example usage in `workloads/` benchmarks

### Scheduler Types

#### C Schedulers (`scx/scheds/c/`)
- Written in C with BPF
- Compiled using clang with BPF target
- Examples: scx_simple, scx_central, scx_flatcg

#### Rust Schedulers (`scx/scheds/rust/`)
- Written in Rust using scx_utils and libbpf-rs
- More complex schedulers with rich logic
- Examples: scx_rusty, scx_lavd, scx_layered
- Built with Cargo, each in its own workspace

### Build Output

- **Binaries**: Compiled schedulers go to `sche_bin/`
- **Tools**: Utilities (scx_loader, scxctl) go to `tools/`
- **Installation**: `make install` copies to `~/.schedcp/scxbin/`

The MCP server embeds binaries from `sche_bin/` during its compilation.

## Autotune Tool (`autotune/`)

AI-powered automatic workload optimizer that combines workload analysis with scheduler selection.

### Directory Structure

```
autotune/
├── src/
│   ├── bin/
│   │   └── cli.rs             # Main CLI entry point
│   ├── daemon.rs              # Command execution and AI integration
│   ├── prompt.rs              # AI prompt generation
│   ├── system_info.rs         # System information collection
│   └── lib.rs                 # Public API
├── Cargo.toml                 # Dependencies
└── README.md                  # Usage documentation
```

### Key Components

#### `cli.rs` - Command-Line Interface
- Subcommands:
  - `run <command>`: Execute and analyze command
  - `cc <command>`: Scheduler optimization workflow
  - `submit <command>`: Submit to daemon (future)
  - `daemon`: Start daemon service (future)
- Argument parsing with clap
- Integration with Claude CLI and MCP server

#### `daemon.rs` - Core Execution Logic
- **Key functions**:
  - `run_command()`: Execute shell command and capture output
  - `get_optimization_suggestions()`: Call Claude for analysis
  - `call_claude_with_prompt()`: Direct Claude CLI interaction
- **Metrics tracked**:
  - Execution duration
  - Exit code
  - stdout/stderr output
  - System resource usage

#### `prompt.rs` - AI Prompt Generation
- Creates specialized prompts for different optimization scenarios:
  - General command optimization
  - Scheduler selection and tuning
  - Workload classification
- Includes system information and workload context
- Structures prompts for actionable AI responses

#### `system_info.rs` - System Context
- Collects system information for AI context:
  - CPU model and core count
  - Memory capacity
  - Kernel version and sched-ext support
  - Available schedulers
- Provides context for AI decision-making

### Integration Points

- **Claude CLI**: Direct integration via command-line
- **MCP Server**: Uses schedcp MCP server for scheduler management
- **Shell Commands**: Executes workloads and captures metrics

### Workflow

The `cc` (scheduler optimization) command performs:
1. Collect system information
2. Execute workload with default scheduler
3. Call AI to analyze workload characteristics
4. AI creates workload profile via MCP
5. AI selects and tests candidate schedulers
6. AI recommends optimal scheduler configuration
7. Apply recommendation and re-run workload

## Workloads (`workloads/`)

Benchmark workloads for testing and evaluating schedulers.

### Directory Structure

```
workloads/
├── basic/                      # Basic scheduling benchmarks
│   ├── schbench/              # Scheduler latency benchmark
│   │   ├── schbench           # Compiled binary
│   │   ├── schbench.c         # Source code
│   │   └── *_bench_start.py   # Benchmark scripts
│   └── scheduler_test/        # Scheduler testing utilities
├── linux-build-bench/          # Linux kernel compilation
│   └── linux/                 # Linux source tree
├── llama.cpp/                  # LLM inference workload
├── cxl-micro/                  # Memory subsystem benchmark
├── processing/                 # Batch processing workloads
├── pytorch/                    # Machine learning workloads
├── redis/                      # Database workloads
├── nginx/                      # Web server workloads
├── faiss/                      # Vector search workloads
└── [other workloads]/
```

### Key Workloads

#### schbench - Scheduler Benchmark
- **Purpose**: Measure scheduler latency and wakeup performance
- **Source**: `basic/schbench/schbench.c`
- **Usage**: Message-passing between threads to stress scheduler
- **Metrics**: Wakeup latency percentiles (50th, 95th, 99th, 99.9th)
- **Benchmark script**: `basic/schbench_test/schbench_bench_start.py`

#### Linux Kernel Build
- **Purpose**: Parallel compilation stress test
- **Location**: `linux-build-bench/linux/`
- **Usage**: `make -C workloads/linux-build-bench/linux -j$(nproc)`
- **Metrics**: Build time (makespan)
- **Characteristics**: CPU-intensive, many short-lived processes

#### llama.cpp - LLM Inference
- **Purpose**: AI inference workload
- **Metrics**: Tokens per second, latency
- **Characteristics**: Memory-intensive, batch processing

#### cxl-micro - Memory Benchmark
- **Purpose**: Memory subsystem performance
- **Metrics**: Bandwidth, latency, NUMA effects
- **Characteristics**: Memory-bound, NUMA-sensitive

### Benchmark Scripts

Python scripts (e.g., `*_bench_start.py`) typically:
1. Use `scheduler_runner.py` to manage schedulers
2. Run workload with different schedulers
3. Collect performance metrics
4. Generate comparison reports (often JSON)
5. Compute statistics and visualizations

Example usage:
```bash
python workloads/basic/scheduler_test/schbench_bench_start.py
```

## Documentation (`document/`)

Research papers, design documents, and development logs.

### Key Documents

- **`2509.01245v2.pdf`**: Research paper "Towards Agentic OS"
- **`sched-agent-design.md`**: Multi-agent architecture design
- **`schedcp-design.md`**: System control plane design
- **`devlog.md`**: Development log and notes
- **`design.png`**: Architecture diagram
- **`schbench-optimize.gif`**: Demo animation

### Design Documents

#### `sched-agent-design.md` - Multi-Agent Framework
Describes the four-agent system:
- **Observation Agent**: Workload analysis and profiling
- **Planning Agent**: Strategy selection and optimization planning
- **Execution Agent**: Code synthesis and safe deployment
- **Learning Agent**: Performance feedback and knowledge curation

See [AI_AGENTS.md](AI_AGENTS.md) for implementation details.

#### `schedcp-design.md` - Control Plane Design
Describes the system architecture:
- **Workload Analysis Engine**: System observation capabilities
- **Scheduler Policy Repository**: Persistent scheduler library
- **Execution Verifier**: Multi-stage validation pipeline

## Data Flow

### End-to-End Workflow

```
User Command (autotune cc "workload")
    ↓
Autotune Tool (autotune/)
    ├─ Executes workload
    ├─ Calls Claude AI
    └─ Sends requests to MCP server
        ↓
MCP Server (mcp/src/)
    ├─ Workload profiling (workload_profile.rs)
    ├─ Scheduler selection (lib.rs, using schedulers.json)
    ├─ Scheduler execution (scheduler_manager.rs)
    ├─ System monitoring (system_monitor.rs)
    └─ Stores results (storage.rs → schedcp_workloads.json)
        ↓
Scheduler Binaries (scheduler/sche_bin/)
    ├─ Embedded in MCP server
    └─ Loaded into kernel via sudo
        ↓
Linux Kernel (sched-ext)
    └─ Executes workload with selected scheduler
        ↓
Performance Metrics
    ├─ Collected by system_monitor
    ├─ Analyzed by AI
    └─ Stored in workload history
```

### File Dependencies

```
schedulers.json (metadata)
    → Embedded in MCP server (mcp/src/lib.rs)
    → Used by AI for scheduler selection

sche_bin/* (binaries)
    → Embedded in MCP server (mcp/src/scheduler_manager.rs)
    → Extracted and executed on demand

schedcp_workloads.json (performance data)
    → Created/updated by MCP server (mcp/src/storage.rs)
    → Used by AI for historical recommendations

~/.schedcp/scxbin/* (installed schedulers)
    → Source for embedding in MCP server
    → Optional: Can run directly without MCP
```

## Build System

### Top-Level Build

```bash
# From project root
make           # Builds all components
```

This delegates to:
1. `scheduler/Makefile` - Builds schedulers
2. `mcp/Cargo.toml` - Builds MCP server (cargo)
3. `autotune/Cargo.toml` - Builds autotune (cargo)

### Component Builds

Each component can be built independently:

```bash
# Schedulers
cd scheduler && make

# MCP server
cd mcp && cargo build --release

# Autotune
cd autotune && cargo build --release
```

### Dependencies

- **Schedulers**: Require clang, libbpf, meson
- **MCP/Autotune**: Require Rust toolchain
- **Runtime**: Require Linux 6.12+ with sched-ext

See `scheduler/Makefile` for dependency installation targets.

## Summary

The SchedCP project is organized around three main components:

1. **MCP Server** (`mcp/`): AI integration layer
   - Manages schedulers programmatically
   - Provides tools for AI assistants
   - Tracks workload performance history

2. **Scheduler System** (`scheduler/`): Build and metadata
   - Compiles sched-ext schedulers
   - Maintains scheduler metadata
   - Provides Python interface

3. **Autotune Tool** (`autotune/`): End-to-end optimizer
   - Analyzes workloads with AI
   - Coordinates scheduler testing
   - Recommends optimal configurations

These components work together to enable AI-powered scheduler optimization for any workload.

For usage instructions, see [USAGE_GUIDE.md](USAGE_GUIDE.md).
For AI agent details, see [AI_AGENTS.md](AI_AGENTS.md).
