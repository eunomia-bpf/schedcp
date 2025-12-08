#!/usr/bin/env python3
"""
FAISS 实验结果分析脚本
用于验证 experiment_analysis.md 中的数据准确性
"""

import json
import os
from pathlib import Path

# 结果目录
RESULTS_DIR = Path(__file__).parent

def load_json(filename):
    """加载 JSON 文件"""
    filepath = RESULTS_DIR / filename
    if filepath.exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def print_separator(title):
    """打印分隔线"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def analyze_single_result(data, name):
    """分析单个结果文件"""
    print(f"\n--- {name} ---")

    config = data.get("config", {})
    print(f"数据集: {config.get('dbname')}")
    print(f"索引类型: {config.get('index_key')}")
    print(f"模式: {'UVM' if config.get('use_uvm') else 'CPU' if config.get('mode') == 'cpu' else 'GPU Device'}")

    # 索引构建时间
    index_add = data.get("index_add", {})
    add_time = index_add.get("total_time", 0)
    total_vectors = index_add.get("total_vectors", 0)
    print(f"\n索引构建:")
    print(f"  总时间: {add_time:.3f}s")
    print(f"  向量数: {total_vectors:,}")
    if add_time > 0:
        throughput = total_vectors / add_time
        print(f"  吞吐量: {throughput/1e6:.2f}M 向量/秒")

    # 搜索结果
    search_results = data.get("search", [])
    print(f"\n搜索结果:")
    for sr in search_results:
        nprobe = sr.get("nprobe")
        search_time = sr.get("search_time", 0)
        qps = sr.get("qps", 0)
        recall = sr.get("recall", {})
        r1 = recall.get("1-R@1", 0)
        r10 = recall.get("1-R@10", 0)
        print(f"  nprobe={nprobe:2d}: 时间={search_time:8.3f}s, QPS={qps:12.2f}, Recall@1={r1:.4f}, Recall@10={r10:.4f}")

    return {
        "add_time": add_time,
        "total_vectors": total_vectors,
        "search": search_results
    }

def compare_results(data1, data2, name1, name2):
    """比较两个结果"""
    print(f"\n比较: {name1} vs {name2}")

    # 索引构建时间比较
    add1 = data1["add_time"]
    add2 = data2["add_time"]
    if add1 > 0 and add2 > 0:
        improvement = (add1 - add2) / add1 * 100
        speedup = add1 / add2
        print(f"\n索引构建时间:")
        print(f"  {name1}: {add1:.3f}s")
        print(f"  {name2}: {add2:.3f}s")
        print(f"  改进: {improvement:.1f}% (加速比: {speedup:.2f}x)")

    # 搜索性能比较
    print(f"\n搜索性能比较:")
    print(f"{'nprobe':>8} | {'时间1':>10} | {'时间2':>10} | {'改进%':>8} | {'QPS1':>12} | {'QPS2':>12} | {'QPS改进%':>10}")
    print("-" * 85)

    search1 = {s["nprobe"]: s for s in data1["search"]}
    search2 = {s["nprobe"]: s for s in data2["search"]}

    for nprobe in sorted(search1.keys()):
        if nprobe in search2:
            t1 = search1[nprobe]["search_time"]
            t2 = search2[nprobe]["search_time"]
            q1 = search1[nprobe]["qps"]
            q2 = search2[nprobe]["qps"]

            time_imp = (t1 - t2) / t1 * 100 if t1 > 0 else 0
            qps_imp = (q2 - q1) / q1 * 100 if q1 > 0 else 0

            print(f"{nprobe:>8} | {t1:>10.3f} | {t2:>10.3f} | {time_imp:>7.1f}% | {q1:>12.2f} | {q2:>12.2f} | {qps_imp:>9.1f}%")

def main():
    print("=" * 70)
    print(" FAISS 实验结果分析 - 数据准确性验证")
    print("=" * 70)

    # 加载所有结果文件
    files = {
        "SIFT20M_GPU": "SIFT20M_IVF4096_Flat_baseline.json",
        "SIFT20M_CPU": "SIFT20M_IVF4096_Flat_cpu.json",
        "SIFT50M_UVM_baseline": "SIFT50M_IVF4096_Flat_uvm_baseline.json",
        "SIFT50M_UVM_prefetch": "SIFT50M_IVF4096_Flat_uvm_prefetch_adaptive_tree.json",
        "SIFT100M_UVM_baseline": "SIFT100M_IVF4096_Flat_uvm_baseline.json",
        "SIFT100M_UVM_prefetch": "SIFT100M_IVF4096_Flat_uvm_prefetch_adaptive_tree.json",
    }

    results = {}
    for name, filename in files.items():
        data = load_json(filename)
        if data:
            results[name] = data
            print(f"✓ 加载: {filename}")
        else:
            print(f"✗ 未找到: {filename}")

    # =========================================================================
    print_separator("1. SIFT20M: CPU vs GPU 设备内存对比")
    # =========================================================================

    if "SIFT20M_CPU" in results and "SIFT20M_GPU" in results:
        cpu_data = analyze_single_result(results["SIFT20M_CPU"], "SIFT20M CPU")
        gpu_data = analyze_single_result(results["SIFT20M_GPU"], "SIFT20M GPU Device")

        print("\n" + "-" * 50)
        print("性能对比:")

        # 索引构建加速比
        cpu_add = cpu_data["add_time"]
        gpu_add = gpu_data["add_time"]
        add_speedup = cpu_add / gpu_add
        print(f"\n索引构建加速比: {add_speedup:.1f}x (CPU: {cpu_add:.2f}s, GPU: {gpu_add:.2f}s)")

        # 搜索加速比
        print("\n搜索加速比:")
        cpu_search = {s["nprobe"]: s for s in cpu_data["search"]}
        gpu_search = {s["nprobe"]: s for s in gpu_data["search"]}

        for nprobe in sorted(cpu_search.keys()):
            if nprobe in gpu_search:
                cpu_time = cpu_search[nprobe]["search_time"]
                gpu_time = gpu_search[nprobe]["search_time"]
                cpu_qps = cpu_search[nprobe]["qps"]
                gpu_qps = gpu_search[nprobe]["qps"]
                speedup = cpu_time / gpu_time
                qps_ratio = gpu_qps / cpu_qps
                print(f"  nprobe={nprobe:2d}: 时间加速={speedup:.1f}x, QPS比={qps_ratio:.1f}x (CPU: {cpu_qps:.0f}, GPU: {gpu_qps:.0f})")

    # =========================================================================
    print_separator("2. SIFT50M: UVM Baseline vs UVM + Prefetch")
    # =========================================================================

    if "SIFT50M_UVM_baseline" in results and "SIFT50M_UVM_prefetch" in results:
        baseline_data = analyze_single_result(results["SIFT50M_UVM_baseline"], "SIFT50M UVM Baseline")
        prefetch_data = analyze_single_result(results["SIFT50M_UVM_prefetch"], "SIFT50M UVM + Prefetch")

        compare_results(baseline_data, prefetch_data, "Baseline", "Prefetch")

    # =========================================================================
    print_separator("3. SIFT100M: UVM Baseline vs UVM + Prefetch")
    # =========================================================================

    if "SIFT100M_UVM_baseline" in results and "SIFT100M_UVM_prefetch" in results:
        baseline_data = analyze_single_result(results["SIFT100M_UVM_baseline"], "SIFT100M UVM Baseline")
        prefetch_data = analyze_single_result(results["SIFT100M_UVM_prefetch"], "SIFT100M UVM + Prefetch")

        compare_results(baseline_data, prefetch_data, "Baseline", "Prefetch")

    # =========================================================================
    print_separator("4. 汇总表格")
    # =========================================================================

    print("\n索引构建时间汇总:")
    print(f"{'配置':<40} | {'时间(s)':>10} | {'吞吐量(M vec/s)':>18}")
    print("-" * 75)

    for name, data in results.items():
        add_time = data.get("index_add", {}).get("total_time", 0)
        total_vec = data.get("index_add", {}).get("total_vectors", 0)
        throughput = total_vec / add_time / 1e6 if add_time > 0 else 0
        print(f"{name:<40} | {add_time:>10.2f} | {throughput:>18.2f}")

    print("\n搜索性能汇总 (nprobe=16):")
    print(f"{'配置':<40} | {'时间(s)':>10} | {'QPS':>12} | {'Recall@1':>10}")
    print("-" * 80)

    for name, data in results.items():
        for sr in data.get("search", []):
            if sr.get("nprobe") == 16:
                search_time = sr.get("search_time", 0)
                qps = sr.get("qps", 0)
                recall = sr.get("recall", {}).get("1-R@1", 0)
                print(f"{name:<40} | {search_time:>10.3f} | {qps:>12.2f} | {recall:>10.4f}")

    # =========================================================================
    print_separator("5. 关键指标验证")
    # =========================================================================

    print("\n文档声明 vs 实际数据:")

    # SIFT20M CPU vs GPU
    if "SIFT20M_CPU" in results and "SIFT20M_GPU" in results:
        cpu_add = results["SIFT20M_CPU"]["index_add"]["total_time"]
        gpu_add = results["SIFT20M_GPU"]["index_add"]["total_time"]

        cpu_search_16 = next((s for s in results["SIFT20M_CPU"]["search"] if s["nprobe"] == 16), {})
        gpu_search_16 = next((s for s in results["SIFT20M_GPU"]["search"] if s["nprobe"] == 16), {})

        print(f"\n[SIFT20M CPU vs GPU]")
        print(f"  索引构建加速比: 文档=21.0x, 实际={cpu_add/gpu_add:.1f}x")
        print(f"  搜索加速比(nprobe=16): 文档=29.5x, 实际={cpu_search_16['search_time']/gpu_search_16['search_time']:.1f}x")
        print(f"  QPS比(nprobe=16): 文档=29.5x, 实际={gpu_search_16['qps']/cpu_search_16['qps']:.1f}x")

    # SIFT50M UVM
    if "SIFT50M_UVM_baseline" in results and "SIFT50M_UVM_prefetch" in results:
        base_add = results["SIFT50M_UVM_baseline"]["index_add"]["total_time"]
        pref_add = results["SIFT50M_UVM_prefetch"]["index_add"]["total_time"]
        improvement = (base_add - pref_add) / base_add * 100

        print(f"\n[SIFT50M UVM Prefetch 优化]")
        print(f"  索引构建时间改进: 文档=21.4%, 实际={improvement:.1f}%")
        print(f"  Baseline: {base_add:.2f}s, Prefetch: {pref_add:.2f}s")

    # SIFT100M UVM
    if "SIFT100M_UVM_baseline" in results and "SIFT100M_UVM_prefetch" in results:
        base_add = results["SIFT100M_UVM_baseline"]["index_add"]["total_time"]
        pref_add = results["SIFT100M_UVM_prefetch"]["index_add"]["total_time"]
        improvement = (base_add - pref_add) / base_add * 100

        base_search_16 = next((s for s in results["SIFT100M_UVM_baseline"]["search"] if s["nprobe"] == 16), {})
        pref_search_16 = next((s for s in results["SIFT100M_UVM_prefetch"]["search"] if s["nprobe"] == 16), {})

        search_improvement = (base_search_16["search_time"] - pref_search_16["search_time"]) / base_search_16["search_time"] * 100
        qps_improvement = (pref_search_16["qps"] - base_search_16["qps"]) / base_search_16["qps"] * 100

        print(f"\n[SIFT100M UVM Prefetch 优化]")
        print(f"  索引构建时间改进: 文档=28.9%, 实际={improvement:.1f}%")
        print(f"  Baseline: {base_add:.2f}s, Prefetch: {pref_add:.2f}s")
        print(f"  搜索时间改进(nprobe=16): 文档=10.5%, 实际={search_improvement:.1f}%")
        print(f"  QPS改进(nprobe=16): 文档=11.7%, 实际={qps_improvement:.1f}%")

    print("\n" + "=" * 70)
    print(" 分析完成")
    print("=" * 70)

if __name__ == "__main__":
    main()
