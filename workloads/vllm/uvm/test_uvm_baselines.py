#!/usr/bin/env python3
"""
Test script for UVM baselines comparison.

Tests three configurations:
1. CPU Offload baseline (cpu-offload-gb 8)
2. UVM baseline (VLLM_USE_UVM=1)
3. LMCache baseline (KV cache offloading with LMCache)

Server directories:
- CPU Offload & UVM: ~/workspace/vllm
- LMCache: /home/yunwei37/workspace/gpu/LMCache

Benchmark runs from: ~/workspace/gpu/schedcp/workloads/vllm
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Directories
VLLM_SERVER_DIR = os.path.expanduser("~/workspace/vllm")
LMCACHE_SERVER_DIR = "/home/yunwei37/workspace/gpu/LMCache"
BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = "/home/yunwei37/workspace/gpu/schedcp/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"

# Model configuration
MODEL = "Qwen/Qwen3-30B-A3B-FP8"

# Server configurations
BASELINE_CONFIGS = {
    "cpu_offload": {
        "name": "CPU Offload (8GB)",
        "server_dir": VLLM_SERVER_DIR,
        "server_cmd": f"uv run vllm serve {MODEL} --enforce-eager --cpu-offload-gb 8",
        "env": {},
    },
    "uvm_baseline": {
        "name": "UVM Baseline",
        "server_dir": VLLM_SERVER_DIR,
        "server_cmd": f"uv run vllm serve {MODEL} --enforce-eager",
        "env": {"VLLM_USE_UVM": "1"},
    },
    "lmcache": {
        "name": "LMCache",
        "server_dir": LMCACHE_SERVER_DIR,
        "server_cmd": (
            f".venv/bin/vllm serve {MODEL} "
            "--max-model-len 2048 "
            "--enforce-eager "
            "--kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' "
            "--cpu-offload-gb 4"
        ),
        "env": {
            "LMCACHE_LOCAL_CPU": "True",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": "8.0",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        },
    },
}


class UVMBaselineTester:
    def __init__(
        self,
        bench_args: str = None,
        output_dir: str = "results",
        server_startup_timeout: int = 600,
        server_ready_check_interval: int = 5,
    ):
        # 默认 bench 参数
        if bench_args is None:
            bench_args = (
                f"--model {MODEL} "
                f"--dataset-name sharegpt "
                f"--num-prompts 100 "
                f"--dataset-path {DATASET_PATH}"
            )
        self.bench_args = bench_args
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建日志目录
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.server_startup_timeout = server_startup_timeout
        self.server_ready_check_interval = server_ready_check_interval
        self.results = {}
        self.server_process = None
        self.server_log_file = None

    def get_bench_cmd(self) -> str:
        """Get benchmark command."""
        return f"uv run vllm bench serve {self.bench_args}"

    def parse_benchmark_output(self, output: str) -> dict:
        """Parse vLLM benchmark output and extract metrics."""
        metrics = {}

        patterns = {
            "successful_requests": r"Successful requests:\s+(\d+)",
            "benchmark_duration": r"Benchmark duration \(s\):\s+([\d.]+)",
            "total_input_tokens": r"Total input tokens:\s+(\d+)",
            "total_generated_tokens": r"Total generated tokens:\s+(\d+)",
            "request_throughput": r"Request throughput \(req/s\):\s+([\d.]+)",
            "output_token_throughput": r"Output token throughput \(tok/s\):\s+([\d.]+)",
            "peak_output_token_throughput": r"Peak output token throughput \(tok/s\):\s+([\d.]+)",
            "peak_concurrent_requests": r"Peak concurrent requests:\s+([\d.]+)",
            "total_token_throughput": r"Total Token throughput \(tok/s\):\s+([\d.]+)",
            "mean_ttft": r"Mean TTFT \(ms\):\s+([\d.]+)",
            "median_ttft": r"Median TTFT \(ms\):\s+([\d.]+)",
            "p99_ttft": r"P99 TTFT \(ms\):\s+([\d.]+)",
            "mean_tpot": r"Mean TPOT \(ms\):\s+([\d.]+)",
            "median_tpot": r"Median TPOT \(ms\):\s+([\d.]+)",
            "p99_tpot": r"P99 TPOT \(ms\):\s+([\d.]+)",
            "mean_itl": r"Mean ITL \(ms\):\s+([\d.]+)",
            "median_itl": r"Median ITL \(ms\):\s+([\d.]+)",
            "p99_itl": r"P99 ITL \(ms\):\s+([\d.]+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, output)
            if match:
                value = match.group(1)
                if "." in value:
                    metrics[key] = float(value)
                else:
                    metrics[key] = int(value)

        return metrics

    def wait_for_server(self, host: str = "127.0.0.1", port: int = 8000) -> bool:
        """Wait for vLLM server to be ready."""
        import socket

        start_time = time.time()
        print(f"Waiting for server at {host}:{port}...")

        while time.time() - start_time < self.server_startup_timeout:
            # Check if server process is still running
            if self.server_process and self.server_process.poll() is not None:
                print("Server process terminated unexpectedly!")
                return False

            try:
                # Try to connect to the server
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((host, port))
                sock.close()

                if result == 0:
                    # Additional check - try a simple HTTP request
                    try:
                        import urllib.request
                        req = urllib.request.Request(f"http://{host}:{port}/health")
                        with urllib.request.urlopen(req, timeout=5) as response:
                            if response.status == 200:
                                print("Server is ready!")
                                return True
                    except Exception:
                        # Health endpoint might not be available, but port is open
                        print("Server port is open, waiting for full initialization...")

            except Exception:
                pass

            elapsed = int(time.time() - start_time)
            print(f"  Waiting... ({elapsed}s / {self.server_startup_timeout}s)")
            time.sleep(self.server_ready_check_interval)

        print("Server startup timeout!")
        return False

    def start_server(self, config_name: str) -> bool:
        """Start vLLM server with given configuration."""
        config = BASELINE_CONFIGS[config_name]
        print(f"\n{'='*60}")
        print(f"Starting server: {config['name']}")
        print(f"Directory: {config['server_dir']}")
        print(f"Command: {config['server_cmd']}")
        if config["env"]:
            print(f"Environment: {config['env']}")
        print("=" * 60)

        # Prepare environment
        env = os.environ.copy()
        env.update(config["env"])

        # 创建 server 日志文件
        server_log_path = self.log_dir / f"{self.run_timestamp}_{config_name}_server.log"
        self.server_log_file = open(server_log_path, "w")
        print(f"Server log: {server_log_path}")

        # Start server process
        try:
            self.server_process = subprocess.Popen(
                config["server_cmd"],
                shell=True,
                cwd=config["server_dir"],
                env=env,
                stdout=self.server_log_file,
                stderr=subprocess.STDOUT,
                text=True,
                preexec_fn=os.setsid,  # Create new process group for clean termination
            )

            # Wait for server to be ready
            if not self.wait_for_server():
                self.stop_server()
                return False

            return True

        except Exception as e:
            print(f"Failed to start server: {e}")
            self.stop_server()
            return False

    def stop_server(self):
        """Stop vLLM server."""
        if self.server_process:
            print("Stopping server...")
            try:
                # Kill the entire process group
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                time.sleep(2)

                # Force kill if still running
                if self.server_process.poll() is None:
                    os.killpg(os.getpgid(self.server_process.pid), signal.SIGKILL)
                    time.sleep(1)

            except Exception as e:
                print(f"Warning during server stop: {e}")

            self.server_process = None

        # 关闭 server 日志文件
        if self.server_log_file:
            try:
                self.server_log_file.close()
            except Exception:
                pass
            self.server_log_file = None

        # Additional cleanup - kill any remaining vllm processes
        try:
            subprocess.run(
                ["pkill", "-f", "vllm serve"],
                capture_output=True,
                check=False,
            )
            time.sleep(1)
        except Exception:
            pass

    def run_benchmark(self, config_name: str) -> dict:
        """Run benchmark for a specific configuration."""
        config = BASELINE_CONFIGS[config_name]
        print(f"\n{'='*60}")
        print(f"Running benchmark: {config['name']}")
        print("=" * 60)

        # Start server
        if not self.start_server(config_name):
            return {"success": False, "error": "Failed to start server"}

        # 创建 client 日志文件路径
        client_log_path = self.log_dir / f"{self.run_timestamp}_{config_name}_client.log"

        try:
            # Run benchmark
            bench_cmd = self.get_bench_cmd()
            print(f"\nBenchmark command: {bench_cmd}")
            print(f"Client log: {client_log_path}")
            print("-" * 60)

            start_time = time.time()
            result = subprocess.run(
                bench_cmd,
                shell=True,
                cwd=BENCH_DIR,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minutes timeout
            )
            end_time = time.time()

            # 保存 client 日志
            with open(client_log_path, "w") as f:
                f.write(f"Command: {bench_cmd}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write("=" * 60 + "\n")
                f.write("STDOUT:\n")
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n" + "=" * 60 + "\n")
                    f.write("STDERR:\n")
                    f.write(result.stderr)

            # Print output
            print(result.stdout)
            if result.stderr:
                print(f"STDERR: {result.stderr}", file=sys.stderr)

            if result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Benchmark failed with return code {result.returncode}",
                    "client_log": str(client_log_path),
                }

            # Parse metrics
            metrics = self.parse_benchmark_output(result.stdout)
            metrics["success"] = True
            metrics["config_name"] = config_name
            metrics["config_display_name"] = config["name"]
            metrics["total_time"] = end_time - start_time
            metrics["bench_args"] = self.bench_args
            metrics["timestamp"] = datetime.now().isoformat()
            metrics["client_log"] = str(client_log_path)
            metrics["server_log"] = str(self.log_dir / f"{self.run_timestamp}_{config_name}_server.log")

            return metrics

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Benchmark timeout", "client_log": str(client_log_path)}
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            self.stop_server()

    def run_all_tests(self, baselines: list = None):
        """Run all baseline tests."""
        if baselines is None:
            baselines = list(BASELINE_CONFIGS.keys())

        print("\n" + "=" * 60)
        print("UVM Baseline Comparison Test")
        print("=" * 60)
        print(f"Baselines to test: {baselines}")
        print(f"Bench args: {self.bench_args}")
        print(f"Output directory: {self.output_dir}")
        print(f"Log directory: {self.log_dir}")
        print("=" * 60)

        for baseline in baselines:
            if baseline not in BASELINE_CONFIGS:
                print(f"Unknown baseline: {baseline}, skipping...")
                continue

            result = self.run_benchmark(baseline)
            self.results[baseline] = result

            if result.get("success"):
                print(f"\n{baseline} completed successfully!")
                print(f"  Mean TTFT: {result.get('mean_ttft', 'N/A')} ms")
                print(f"  Mean TPOT: {result.get('mean_tpot', 'N/A')} ms")
                print(f"  Request throughput: {result.get('request_throughput', 'N/A')} req/s")
            else:
                print(f"\n{baseline} FAILED: {result.get('error', 'Unknown error')}")

        # Save results
        self.save_results()

        # Print summary
        self.print_summary()

    def save_results(self):
        """Save results to JSON file."""
        output_file = self.output_dir / f"uvm_baseline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        output_data = {
            "test_info": {
                "timestamp": datetime.now().isoformat(),
                "bench_args": self.bench_args,
                "model": MODEL,
            },
            "results": self.results,
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {output_file}")

        # Also save a latest symlink
        latest_file = self.output_dir / "uvm_baseline_results_latest.json"
        if latest_file.exists():
            latest_file.unlink()
        latest_file.symlink_to(output_file.name)

    def print_summary(self):
        """Print summary of all results."""
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        successful_results = {k: v for k, v in self.results.items() if v.get("success")}

        if not successful_results:
            print("No successful tests!")
            return

        # Print table header
        print(f"{'Baseline':<20} {'TTFT (ms)':<15} {'TPOT (ms)':<15} {'Throughput':<12}")
        print("-" * 62)

        # Sort by mean TTFT
        for name, result in sorted(
            successful_results.items(),
            key=lambda x: x[1].get("mean_ttft", float("inf")),
        ):
            display_name = BASELINE_CONFIGS[name]["name"]
            ttft = f"{result.get('mean_ttft', 0):.2f}"
            tpot = f"{result.get('mean_tpot', 0):.2f}"
            throughput = f"{result.get('request_throughput', 0):.2f} req/s"
            print(f"{display_name:<20} {ttft:<15} {tpot:<15} {throughput:<12}")

        print("-" * 62)

        # Print detailed latency comparison
        print("\nDetailed Latency Comparison:")
        print("-" * 62)

        for name, result in successful_results.items():
            display_name = BASELINE_CONFIGS[name]["name"]
            print(f"\n{display_name}:")
            print(f"  TTFT - Mean: {result.get('mean_ttft', 'N/A'):.2f} ms, "
                  f"Median: {result.get('median_ttft', 'N/A'):.2f} ms, "
                  f"P99: {result.get('p99_ttft', 'N/A'):.2f} ms")
            print(f"  TPOT - Mean: {result.get('mean_tpot', 'N/A'):.2f} ms, "
                  f"Median: {result.get('median_tpot', 'N/A'):.2f} ms, "
                  f"P99: {result.get('p99_tpot', 'N/A'):.2f} ms")
            print(f"  ITL  - Mean: {result.get('mean_itl', 'N/A'):.2f} ms, "
                  f"Median: {result.get('median_itl', 'N/A'):.2f} ms, "
                  f"P99: {result.get('p99_itl', 'N/A'):.2f} ms")


def main():
    parser = argparse.ArgumentParser(
        description="Run UVM baseline comparison tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all baselines with default settings
  python test_uvm_baselines.py

  # Run specific baselines
  python test_uvm_baselines.py --baselines cpu_offload uvm_baseline

  # Custom benchmark arguments (passed directly to vllm bench serve)
  python test_uvm_baselines.py --bench-args "--model Qwen/Qwen3-30B-A3B-FP8 --dataset-name sharegpt --num-prompts 50 --dataset-path /path/to/dataset.json"

  # Run only LMCache test with custom args
  python test_uvm_baselines.py --baselines lmcache --bench-args "--model Qwen/Qwen3-30B-A3B-FP8 --num-prompts 20 --dataset-name sharegpt --dataset-path /path/to/dataset.json"

Available baselines:
  cpu_offload  - CPU Offload with 8GB offload
  uvm_baseline - UVM baseline (VLLM_USE_UVM=1)
  lmcache      - LMCache with KV cache offloading
        """,
    )

    parser.add_argument(
        "--bench-args",
        type=str,
        default=None,
        help="Arguments to pass to vllm bench serve (default: --model Qwen/Qwen3-30B-A3B-FP8 --dataset-name sharegpt --num-prompts 100 --dataset-path ...)",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=list(BASELINE_CONFIGS.keys()),
        default=None,
        help="Specific baselines to test (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)",
    )
    parser.add_argument(
        "--server-timeout",
        type=int,
        default=600,
        help="Server startup timeout in seconds (default: 600)",
    )

    args = parser.parse_args()

    # Run tests
    tester = UVMBaselineTester(
        bench_args=args.bench_args,
        output_dir=args.output_dir,
        server_startup_timeout=args.server_timeout,
    )

    try:
        tester.run_all_tests(baselines=args.baselines)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        tester.stop_server()
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        tester.stop_server()
        sys.exit(1)


if __name__ == "__main__":
    main()
