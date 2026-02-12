# AI Agents in SchedCP

This document explains how AI agents are implemented in SchedCP and how they work together to optimize Linux kernel schedulers. While the conceptual design describes four specialized agents, the implementation uses a flexible approach that achieves the same goals.

## Table of Contents

- [Conceptual Design vs Implementation](#conceptual-design-vs-implementation)
- [Agent Roles and Implementation](#agent-roles-and-implementation)
- [How Agents Work](#how-agents-work)
- [MCP Tools and Agent Capabilities](#mcp-tools-and-agent-capabilities)
- [AI Integration Points](#ai-integration-points)
- [Example Agent Workflows](#example-agent-workflows)

## Conceptual Design vs Implementation

### Conceptual Design (from Research Paper)

The research paper "Towards Agentic OS" (document/sched-agent-design.md) describes a multi-agent framework with four specialized agents:

1. **Observation & Analysis Agent**: Workload profiling and system sensing
2. **Planning Agent**: Strategy selection and optimization planning
3. **Execution Agent**: Code synthesis and safe deployment
4. **Learning Agent**: Performance feedback and knowledge curation

### Actual Implementation

SchedCP implements these agent capabilities through:

1. **AI Assistant (Claude/LLM)**: Provides reasoning and decision-making
2. **MCP Server Tools**: Expose system capabilities to AI
3. **Autotune Orchestrator**: Coordinates end-to-end workflows
4. **Storage System**: Maintains persistent knowledge

Rather than having separate code files for each agent, the agent behaviors emerge from:
- The AI's natural language understanding and reasoning
- The tools available through the MCP protocol
- The prompts and context provided by autotune
- The structured data in schedulers.json and workload profiles

This design is more flexible and maintainable while achieving the same functionality.

## Agent Roles and Implementation

### 1. Observation & Analysis Agent

**Conceptual Role**: Performs deep semantic analysis of workloads and synthesizes findings into structured Workload Profiles.

**Implementation**:

#### Tools (MCP Server)
- **`system_monitor`** (`mcp/src/system_monitor.rs`):
  - Collects CPU utilization from `/proc/stat`
  - Tracks memory usage from `/proc/meminfo`
  - Monitors scheduler statistics from `/proc/schedstat`
  - Samples every 1 second during execution
  
- **`workload` (create command)** (`mcp/src/workload_profile.rs`):
  - Creates workload profiles with natural language descriptions
  - Stores classification (latency-sensitive, throughput-focused, etc.)
  - Maintains unique identifier for tracking

- **`get_execution_status`** (`mcp/src/scheduler_manager.rs`):
  - Retrieves output from running schedulers
  - Captures performance metrics

#### Command Execution (Autotune)
- **`run_command()`** (`autotune/src/daemon.rs`):
  - Executes workload commands
  - Captures stdout/stderr
  - Measures execution duration
  - Records exit codes

#### AI Capabilities
The AI assistant (Claude) performs:
- Analysis of command syntax and arguments
- Inference of workload characteristics from command patterns
- Synthesis of natural language workload descriptions
- Classification into workload types

**Example Observation Workflow**:
```
1. User: autotune cc "make -j$(nproc)"
2. Autotune executes command and collects metrics
3. AI analyzes: "parallel compilation, CPU-intensive, short-lived processes"
4. AI calls: workload(create) with description
5. Workload profile created with classification
```

**Where the "Agent" Lives**:
- Reasoning: AI assistant (Claude)
- Tools: `mcp/src/system_monitor.rs`, `mcp/src/workload_profile.rs`
- Orchestration: `autotune/src/daemon.rs`
- Data: `schedcp_workloads.json`

### 2. Planning Agent

**Conceptual Role**: Synthesizes concrete optimization plans using Workload Profiles to guide strategy.

**Implementation**:

#### Knowledge Base
- **`schedulers.json`** (`scheduler/schedulers.json`):
  - Comprehensive metadata for all schedulers
  - Algorithm descriptions and characteristics
  - Use case classifications
  - Tuning parameter specifications with ranges
  - Production readiness indicators

- **`list_schedulers`** (`mcp/src/lib.rs`):
  - Queries scheduler metadata
  - Filters by production readiness
  - Searches by name or characteristics

#### AI Capabilities
The AI assistant performs:
- Semantic matching between workload characteristics and scheduler metadata
- Strategy selection (configure existing vs. create custom)
- Parameter selection based on workload requirements
- Multi-scheduler testing planning

**Example Planning Workflow**:
```
1. AI receives workload profile: "latency-sensitive gaming workload"
2. AI calls: list_schedulers(production_ready=true)
3. AI analyzes scheduler metadata:
   - scx_bpfland: use_cases includes "gaming" ✓
   - scx_lavd: characteristics include "latency-aware" ✓
   - scx_rusty: general-purpose but less focused
4. AI creates test plan:
   - Test scx_bpfland with slice_us=5000 (low latency)
   - Test scx_lavd with default params
   - Compare against default scheduler
5. AI generates commands for execution agent
```

**Where the "Agent" Lives**:
- Reasoning: AI assistant (Claude)
- Knowledge: `scheduler/schedulers.json` (embedded in MCP server)
- Tools: `mcp/src/lib.rs` (list_schedulers)
- Prompts: `autotune/src/prompt.rs` (guides AI reasoning)

### 3. Execution Agent

**Conceptual Role**: Validates and deploys proposed policies with strict safety guarantees.

**Implementation**:

#### Scheduler Management
- **`run_scheduler`** (`mcp/src/scheduler_manager.rs`):
  - Loads scheduler binaries (embedded resources)
  - Executes with sudo privileges
  - Captures stdout/stderr
  - Tracks process status

- **`stop_scheduler`** (`mcp/src/scheduler_manager.rs`):
  - Gracefully terminates schedulers
  - Restores default scheduler automatically
  - Cleans up resources

- **`create_and_verify_scheduler`** (`mcp/src/scheduler_generator.rs`):
  - Compiles custom BPF schedulers from source
  - Verifies by loading in kernel for 10 seconds
  - Provides compilation error feedback
  - Stores compiled schedulers in `mcp/new_sched/`

#### Safety Mechanisms

**Static Validation**:
- Source code checked for required BPF operations
- Compilation with strict BPF target flags
- Clang verification during compilation

**Dynamic Validation**:
- 10-second kernel verification test
- Automatic rollback on verification failure
- Detailed error reporting

**Production Safeguards**:
- All schedulers require explicit sudo
- Ctrl+C immediately restores default scheduler
- Process monitoring prevents zombie processes
- Comprehensive logging to `schedcp.log`

**Example Execution Workflow**:
```
1. AI decides to test scx_bpfland with custom parameters
2. AI calls: run_scheduler("scx_bpfland", ["--slice-us", "5000"])
3. MCP server:
   - Loads embedded scx_bpfland binary
   - Executes with sudo using SCHEDCP_SUDO_PASSWORD
   - Creates execution record with unique ID
   - Monitors process
4. Workload runs under scheduler
5. AI calls: stop_scheduler(execution_id)
6. MCP server terminates scheduler, restores default
```

**Where the "Agent" Lives**:
- Reasoning: AI assistant (Claude)
- Execution: `mcp/src/scheduler_manager.rs`, `mcp/src/scheduler_generator.rs`
- Safety: Built into BPF verifier and MCP server validation
- Process management: `mcp/lib/process_manager/`

### 4. Learning Agent

**Conceptual Role**: Translates action outcomes into durable, reusable knowledge for system improvement.

**Implementation**:

#### Knowledge Storage
- **`workload` (add_history command)** (`mcp/src/storage.rs`):
  - Adds execution results to workload profiles
  - Stores: scheduler name, parameters, metrics, timestamp
  - Persists to `schedcp_workloads.json`

- **`workload` (get_history command)** (`mcp/src/storage.rs`):
  - Retrieves historical performance data
  - Enables comparison across schedulers
  - Supports data-driven recommendations

#### Learning Mechanisms

**Short-term (In-Context) Learning**:
- AI maintains conversation context
- Compares results across test runs in same session
- Adjusts testing strategy based on intermediate results
- Example: "scx_bpfland performed poorly, trying scx_lavd with different params"

**Long-term (Persistent) Learning**:
- Workload profiles store historical data
- AI queries history for similar workloads
- Recommendations improve over time with more data
- Example: "For kernel builds, scx_rusty historically performs best"

**Example Learning Workflow**:
```
1. AI completes scheduler testing
2. Results: scx_bpfland reduced latency by 35%
3. AI calls: workload(add_history, workload_id, results)
4. Data stored in schedcp_workloads.json
5. Future session with similar workload:
6. AI calls: workload(get_history, workload_id)
7. AI: "Historical data shows scx_bpfland is optimal for this workload"
8. AI applies best-known configuration immediately
```

**Where the "Agent" Lives**:
- Reasoning: AI assistant (Claude)
- Storage: `mcp/src/storage.rs`, `mcp/src/workload_profile.rs`
- Data: `schedcp_workloads.json` (persistent)
- Analysis: AI's natural language understanding and pattern recognition

## How Agents Work

### Agent Communication

Instead of explicit inter-agent messages, communication happens through:

1. **Structured Data**: Workload profiles, execution results
2. **Shared Context**: AI maintains conversation state
3. **Tool Results**: Each tool returns structured data
4. **Persistent Storage**: `schedcp_workloads.json` acts as shared memory

### Agent Coordination

The **autotune** tool orchestrates agent workflows:

```rust
// autotune/src/daemon.rs
pub fn get_optimization_suggestions(result: &CommandResult) -> Result<String, String> {
    // 1. Create prompt with system context
    let prompt = crate::prompt::create_optimization_prompt(
        &result.command,
        result.duration,
        result.exit_code,
        &result.stdout,
        &result.stderr,
    );
    
    // 2. Call AI assistant
    let claude_output = call_claude_with_prompt(&prompt, false)?;
    
    // 3. AI assistant:
    //    - Analyzes workload (Observation Agent role)
    //    - Queries schedulers (Planning Agent role)
    //    - Runs tests (Execution Agent role)
    //    - Stores results (Learning Agent role)
    //    All through MCP tools
    
    Ok(claude_output)
}
```

The AI assistant automatically sequences agent behaviors based on the task.

### Agent Specialization

While not separate code modules, agents are specialized through:

1. **Tool Design**: Each tool maps to specific agent capabilities
   - `system_monitor` → Observation
   - `list_schedulers` → Planning
   - `run_scheduler` → Execution
   - `workload` (history) → Learning

2. **Prompt Engineering**: Prompts guide AI to specific behaviors
   - Observation prompts: "Analyze this workload..."
   - Planning prompts: "Select optimal schedulers..."
   - Execution prompts: "Test the following configurations..."
   - Learning prompts: "Based on historical data..."

3. **Data Structures**: Structured data shapes agent interactions
   - Workload profiles (Observation → Planning)
   - Execution results (Execution → Learning)
   - Historical data (Learning → Planning)

## MCP Tools and Agent Capabilities

### Tool-to-Agent Mapping

| MCP Tool | Primary Agent | Purpose |
|----------|---------------|---------|
| `system_monitor` | Observation | Collect real-time metrics |
| `workload` (create) | Observation | Create workload profile |
| `list_schedulers` | Planning | Query scheduler metadata |
| `run_scheduler` | Execution | Execute scheduler |
| `stop_scheduler` | Execution | Stop scheduler |
| `get_execution_status` | Execution | Check status |
| `create_and_verify_scheduler` | Execution | Create custom scheduler |
| `workload` (get_history) | Learning | Retrieve historical data |
| `workload` (add_history) | Learning | Store execution results |

### Tool Implementation Details

See `mcp/src/lib.rs` for complete MCP tool implementations. Key excerpts:

```rust
// Observation: System monitoring
async fn system_monitor_tool(args: SystemMonitorArgs) -> Result<String> {
    match args.command.as_str() {
        "start" => system_monitor::start_monitoring(),
        "stop" => system_monitor::stop_and_get_summary(),
        _ => Err("Unknown command")
    }
}

// Planning: Scheduler listing
async fn list_schedulers_tool(args: ListSchedulersArgs) -> Result<Vec<Scheduler>> {
    let schedulers = load_schedulers_from_embedded()?;
    filter_schedulers(schedulers, args.name, args.production_ready)
}

// Execution: Run scheduler
async fn run_scheduler_tool(args: RunSchedulerArgs) -> Result<String> {
    scheduler_manager::create_execution(
        args.name,
        args.args.unwrap_or_default()
    )
}

// Learning: Workload history
async fn workload_tool(args: WorkloadArgs) -> Result<String> {
    match args.command.as_str() {
        "create" => create_workload_profile(args),
        "get_history" => get_workload_history(args.workload_id),
        "add_history" => add_execution_to_history(args),
        _ => Err("Unknown command")
    }
}
```

## AI Integration Points

### Claude Integration

SchedCP integrates with Claude through two mechanisms:

#### 1. Model Context Protocol (MCP)
- **Protocol**: stdio-based JSON-RPC
- **Server**: `mcp/src/main.rs` implements MCP server
- **Client**: Claude Desktop or Claude Code
- **Configuration**: `.mcp.json` in project root
- **Communication**: Bidirectional tool calls and responses

#### 2. Claude CLI (Autotune)
- **Tool**: Command-line Claude client
- **Usage**: `autotune/src/daemon.rs` calls `claude` command
- **Prompts**: Generated by `autotune/src/prompt.rs`
- **Mode**: Interactive or single-shot

### Integration with Other AI Systems

The MCP server is AI-agnostic and can work with any AI system supporting MCP:
- OpenAI GPT models (via MCP clients)
- Local LLMs (via MCP clients)
- Custom AI systems implementing MCP protocol

### Prompt Engineering

Prompts are critical for guiding AI agent behaviors:

```rust
// autotune/src/prompt.rs
pub fn create_optimization_prompt(
    command: &str,
    duration: Duration,
    exit_code: i32,
    stdout: &str,
    stderr: &str,
) -> String {
    format!(
        "Analyze this command execution and provide optimization suggestions:\n\
         Command: {}\n\
         Duration: {:?}\n\
         Exit Code: {}\n\
         Output: {}\n\
         Error: {}\n\n\
         Consider:\n\
         1. Workload characteristics (CPU/memory/IO intensive)\n\
         2. Appropriate kernel scheduler selection\n\
         3. System configuration optimizations\n\
         4. Parallelization opportunities\n\n\
         Use available MCP tools to test scheduler configurations.",
        command, duration, exit_code, stdout, stderr
    )
}
```

This prompt guides the AI to act as the Observation and Planning agents.

## Example Agent Workflows

### Complete Optimization Workflow

```
1. USER ACTION
   $ autotune cc "make -j$(nproc)"

2. OBSERVATION AGENT (AI + autotune)
   - Autotune executes command: make -j$(nproc)
   - Records: duration=45s, exit_code=0, output=<build logs>
   - AI analyzes: "Parallel compilation, CPU-intensive, many processes"
   - AI calls: workload(create, "Linux kernel build", "throughput-focused")
   - Result: workload_id="workload_12345"

3. PLANNING AGENT (AI + MCP)
   - AI calls: list_schedulers(production_ready=true)
   - AI analyzes metadata:
     * scx_rusty: "multi-domain, load balancing" → Good for parallel builds ✓
     * scx_p2dq: "pick-two balancing" → Suitable ✓
     * scx_bpfland: "interactive tasks" → Not optimal for builds ✗
   - AI creates plan:
     a) Test default scheduler (baseline)
     b) Test scx_rusty with default params
     c) Test scx_rusty with optimized params
     d) Test scx_p2dq

4. EXECUTION AGENT (AI + MCP + scheduler_manager)
   Test 1 - Default:
   - AI: No scheduler change
   - Autotune: Runs make, measures 45s
   
   Test 2 - scx_rusty default:
   - AI calls: system_monitor(start)
   - AI calls: run_scheduler("scx_rusty", [])
   - Autotune: Runs make, measures 38s
   - AI calls: get_execution_status(exec_id) → captures output
   - AI calls: stop_scheduler(exec_id)
   - AI calls: system_monitor(stop) → gets CPU/memory stats
   
   Test 3 - scx_rusty optimized:
   - AI calls: run_scheduler("scx_rusty", ["--slice-us-underutil", "20000"])
   - Autotune: Runs make, measures 36s
   - AI calls: stop_scheduler(exec_id)
   
   Test 4 - scx_p2dq:
   - AI calls: run_scheduler("scx_p2dq", [])
   - Autotune: Runs make, measures 40s
   - AI calls: stop_scheduler(exec_id)

5. LEARNING AGENT (AI + storage)
   - AI compiles results:
     * default: 45s
     * scx_rusty (default): 38s (16% improvement)
     * scx_rusty (optimized): 36s (20% improvement) ← BEST
     * scx_p2dq: 40s (11% improvement)
   
   - AI calls: workload(add_history, workload_id="workload_12345", 
                results=[...all test results...])
   
   - Storage system writes to schedcp_workloads.json:
     {
       "workload_12345": {
         "description": "Linux kernel build",
         "classification": "throughput-focused",
         "history": [
           {
             "scheduler": "scx_rusty",
             "args": ["--slice-us-underutil", "20000"],
             "duration": 36.2,
             "improvement": 0.20,
             "timestamp": "2024-..."
           },
           ...
         ]
       }
     }

6. RECOMMENDATION
   - AI summarizes: "Best scheduler: scx_rusty with --slice-us-underutil 20000"
   - AI provides: "20% reduction in build time (45s → 36s)"
   - AI explains: "Load balancing and underutilization optimization improve 
                   parallel compilation performance"
```

### Custom Scheduler Creation Workflow

```
1. OBSERVATION AGENT
   - AI identifies: "Standard schedulers don't meet requirements"
   - AI analyzes: "Need NUMA-aware scheduling for memory-intensive workload"

2. PLANNING AGENT
   - AI searches: list_schedulers() to understand existing approaches
   - AI decides: "Create custom NUMA-aware scheduler based on scx_rusty"
   - AI designs: Algorithm combining NUMA awareness with load balancing

3. EXECUTION AGENT
   - AI generates: BPF C code for custom scheduler
   - AI calls: create_and_verify_scheduler(
       name="scx_numa_aware",
       source_code="<BPF C code>",
       description="NUMA-aware scheduler",
       use_cases=["memory-intensive", "NUMA systems"]
     )
   
   - MCP server (scheduler_generator.rs):
     a) Validates source has required BPF ops
     b) Compiles with clang: scx_numa_aware.bpf.c → scx_numa_aware.bpf.o
     c) Loads in kernel for 10-second verification
     d) Stores in mcp/new_sched/
   
   - Result: "custom:scx_numa_aware" available for use

4. EXECUTION AGENT (Testing)
   - AI calls: run_scheduler("custom:scx_numa_aware", [])
   - Autotune: Runs workload, measures performance
   - AI compares: Custom vs. standard schedulers

5. LEARNING AGENT
   - AI stores: Results in workload history
   - AI contributes: If successful, custom scheduler becomes reusable
   - Future: Other workloads can benefit from custom:scx_numa_aware
```

### Historical Data-Driven Optimization

```
1. NEW WORKLOAD
   - User: autotune cc "make -C myproject -j"

2. OBSERVATION AGENT
   - AI analyzes: "Similar to kernel build workload"
   - AI calls: workload(create, "C++ project build", "throughput-focused")

3. LEARNING AGENT (Retrieval)
   - AI calls: workload(list) → sees "workload_12345" (kernel build)
   - AI calls: workload(get_history, "workload_12345")
   - AI receives: Historical data showing scx_rusty performed best

4. PLANNING AGENT (Informed)
   - AI reasons: "Similar workload, use historical best as starting point"
   - AI creates plan: "Start with scx_rusty optimized params"
   - AI skips: Extensive testing, uses known-good configuration

5. EXECUTION AGENT
   - AI calls: run_scheduler("scx_rusty", ["--slice-us-underutil", "20000"])
   - Autotune: Runs build with optimal scheduler immediately
   - Result: Fast optimization with minimal testing

6. LEARNING AGENT
   - AI calls: workload(add_history) with new results
   - Knowledge base grows: More data for future optimizations
```

## Summary

SchedCP implements AI agent capabilities through:

1. **Flexible AI Integration**: Claude (or other AIs) provide reasoning
2. **Specialized MCP Tools**: Each tool maps to agent capabilities
3. **Structured Data**: Enables agent communication and coordination
4. **Persistent Knowledge**: Learning accumulates in `schedcp_workloads.json`
5. **Orchestration**: Autotune coordinates end-to-end workflows

The design achieves the four-agent architecture's goals while maintaining flexibility and simplicity:

- **Observation**: System monitoring + workload profiling
- **Planning**: Scheduler metadata + AI reasoning
- **Execution**: Scheduler management + safety validation
- **Learning**: Persistent storage + historical analysis

Rather than rigid agent boundaries, the system allows the AI to naturally sequence these capabilities based on the task at hand.

For usage examples, see [USAGE_GUIDE.md](USAGE_GUIDE.md).
For project structure, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).
