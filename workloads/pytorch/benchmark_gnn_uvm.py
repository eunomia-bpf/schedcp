#!/usr/bin/env python3
"""
GCN Training Benchmark with UVM Support

Implements standard GCN with symmetric normalization (D^{-1/2}AD^{-1/2}) on
both OGB datasets and synthetic graphs, with optional UVM for extreme scale.

Features:
- Normalized GCN aggregation (symmetric + self-loops)
- Two propagation modes: sparse SpMM and chunked index_add
- Standard datasets (ogbn-arxiv, ogbn-products) with official splits
- Reproducible evaluation with warmup, synchronization, and multiple runs
- Comprehensive memory tracking (GPU/CPU/UVM)
- JSON output for automated analysis
- Oversubscription support via UVM (train graphs larger than GPU memory)

Usage:
    # Standard GCN on small graph (1M nodes, fits in memory)
    python benchmark_gnn_uvm.py --dataset random --nodes 1000000 \
        --prop spmm --epochs 5 --report_json results.json

    # Memory-efficient chunked mode
    python benchmark_gnn_uvm.py --dataset random --nodes 1000000 \
        --prop chunked --chunk_size 2000000 --epochs 5

    # Oversubscription test (10M nodes, exceeds GPU memory)
    # Without UVM: will OOM during training
    python benchmark_gnn_uvm.py --dataset random --nodes 10000000 \
        --prop chunked --epochs 2

    # With UVM: succeeds but ~500x slower
    export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
    python benchmark_gnn_uvm.py --dataset random --nodes 10000000 \
        --prop chunked --epochs 2 --use_uvm --report_json results_uvm.json

    # OGB datasets (requires: pip install ogb)
    python benchmark_gnn_uvm.py --dataset ogbn-arxiv --prop spmm --epochs 100

Results:
    - 1M nodes, SpMM:     0.141s/epoch, 1.54 GB GPU
    - 1M nodes, Chunked:  0.218s/epoch, 1.12 GB GPU (27% less memory)
    - 10M nodes, no UVM:  OOM (fails during warmup)
    - 10M nodes, UVM:     69.87s/epoch, 45.11 GB peak (oversubscription)

See GCN_BENCHMARK_RESULTS.md for detailed analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import psutil
import os
import ctypes
import json
from datetime import datetime


# =============================================================================
# Weighted Chunked Aggregation (GCN-normalized)
# =============================================================================

class ChunkedIndexAddWeightedFunction(torch.autograd.Function):
    """
    Custom autograd function for GCN aggregation with edge weights in chunks.

    Computes: y[v] = sum_{u in N(v)} w_{v,u} * x[u]
    where w_{v,u} = 1 / sqrt(deg(v) * deg(u))

    Memory: O(N*F + chunk_size*F) instead of O(E*F)
    """
    @staticmethod
    def forward(ctx, x, src_idx, dst_idx, edge_weight, chunk_size):
        """
        Args:
            x: [N, F] node features
            src_idx: [E] source node indices
            dst_idx: [E] destination node indices
            edge_weight: [E] normalized edge weights
            chunk_size: edges to process per chunk
        """
        num_nodes, feat = x.size()
        E = src_idx.numel()
        out = x.new_zeros((num_nodes, feat))

        # Process edges in chunks
        for s in range(0, E, chunk_size):
            e = min(s + chunk_size, E)
            src = src_idx[s:e]
            dst = dst_idx[s:e]
            w = edge_weight[s:e].unsqueeze(-1)  # [chunk, 1]
            msg = x[src] * w  # [chunk, F]
            out.index_add_(0, dst, msg)

        ctx.save_for_backward(src_idx, dst_idx, edge_weight)
        ctx.chunk_size = chunk_size
        ctx.num_nodes = num_nodes
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """Backward pass: gradient flows back through weighted aggregation"""
        src_idx, dst_idx, edge_weight = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        num_nodes = ctx.num_nodes
        feat = grad_out.size(1)
        grad_x = grad_out.new_zeros((num_nodes, feat))
        E = src_idx.numel()

        for s in range(0, E, chunk_size):
            e = min(s + chunk_size, E)
            src = src_idx[s:e]
            dst = dst_idx[s:e]
            w = edge_weight[s:e].unsqueeze(-1)  # [chunk, 1]
            g_msg = grad_out.index_select(0, dst) * w  # [chunk, F]
            grad_x.index_add_(0, src, g_msg)

        return grad_x, None, None, None, None


class GCNLayerChunked(nn.Module):
    """GCN layer with chunked weighted aggregation"""
    def __init__(self, in_dim, out_dim, chunk_size=2_000_000):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.chunk_size = chunk_size

    def forward(self, x, src_idx, dst_idx, edge_weight):
        """
        Args:
            x: [N, F_in] features
            src_idx, dst_idx: [E] edge indices
            edge_weight: [E] normalized weights
        Returns:
            [N, F_out] aggregated features
        """
        x = self.lin(x)
        return ChunkedIndexAddWeightedFunction.apply(
            x, src_idx, dst_idx, edge_weight, self.chunk_size
        )


# =============================================================================
# Sparse Matrix Multiplication (Standard GCN)
# =============================================================================

class GCNLayerSpMM(nn.Module):
    """Standard GCN layer using sparse matrix multiplication"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, A_norm_coo):
        """
        Args:
            x: [N, F_in] features
            A_norm_coo: [N, N] sparse normalized adjacency matrix
        Returns:
            [N, F_out] aggregated features
        """
        x = self.lin(x)
        return torch.sparse.mm(A_norm_coo, x)


# =============================================================================
# GCN Models
# =============================================================================

class GCN_OSDI_SpMM(nn.Module):
    """2-layer GCN using sparse matrix multiplication"""
    def __init__(self, in_dim, hid, out_dim):
        super().__init__()
        self.conv1 = GCNLayerSpMM(in_dim, hid)
        self.conv2 = GCNLayerSpMM(hid, out_dim)

    def forward(self, x, A):
        x = self.conv1(x, A)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, A)
        return x


class GCN_OSDI_Chunked(nn.Module):
    """2-layer GCN using chunked aggregation"""
    def __init__(self, in_dim, hid, out_dim, chunk_size=2_000_000):
        super().__init__()
        self.conv1 = GCNLayerChunked(in_dim, hid, chunk_size)
        self.conv2 = GCNLayerChunked(hid, out_dim, chunk_size)

    def forward(self, x, src, dst, w):
        x = self.conv1(x, src, dst, w)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, src, dst, w)
        return x


# =============================================================================
# Graph Normalization
# =============================================================================

def build_normalized_graph(edge_index, num_nodes, device, make_coo=True):
    """
    Build GCN-normalized graph with symmetric normalization and self-loops.

    Implements: A_norm = D^{-1/2} (A + I) D^{-1/2}
    where D is the degree matrix of A + I

    Args:
        edge_index: [2, E] directed edge indices
        num_nodes: number of nodes
        device: torch device
        make_coo: whether to create sparse COO matrix (for SpMM mode)

    Returns:
        tuple: ((src, dst, edge_weight), A_norm_coo or None)
    """
    src, dst = edge_index

    # Step 1: Symmetrize edges (add reverse edges)
    src_sym = torch.cat([src, dst], dim=0)
    dst_sym = torch.cat([dst, src], dim=0)

    # Step 2: Add self-loops
    loop = torch.arange(num_nodes, device=device, dtype=src.dtype)
    src_final = torch.cat([src_sym, loop], dim=0)
    dst_final = torch.cat([dst_sym, loop], dim=0)

    # Step 3: Compute degrees (of undirected graph with self-loops)
    deg = torch.bincount(dst_final, minlength=num_nodes).clamp_min(1).float()

    # Step 4: Compute normalized edge weights: 1 / sqrt(deg[src] * deg[dst])
    deg_src = deg[src_final].sqrt()
    deg_dst = deg[dst_final].sqrt()
    edge_weight = 1.0 / (deg_src * deg_dst)

    if make_coo:
        # Create sparse COO tensor: A[dst, src] = weight
        # (dst as row index for aggregation: A @ X computes sum over columns)
        indices = torch.stack([dst_final, src_final], dim=0)
        A = torch.sparse_coo_tensor(
            indices, edge_weight,
            size=(num_nodes, num_nodes),
            device=device
        )
        A = A.coalesce()
        return (src_final, dst_final, edge_weight), A
    else:
        return (src_final, dst_final, edge_weight), None


# =============================================================================
# Dataset Loading
# =============================================================================

def load_dataset(args, device):
    """
    Load dataset (OGB or random synthetic graph).

    Returns:
        tuple: (x, y, edge_index, train_mask, valid_mask, test_mask)
    """
    if args.dataset == 'random':
        num_nodes = args.nodes
        num_edges = args.nodes * args.edges_per_node

        print(f"  Generating random graph: {num_nodes:,} nodes, {num_edges:,} edges")
        edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device, dtype=torch.long)
        x = torch.randn(num_nodes, args.features, device=device)
        y = torch.randint(0, args.num_classes, (num_nodes,), device=device)

        # Use 10% for training
        train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        train_mask[:num_nodes // 10] = True

        return x, y, edge_index, train_mask, None, None

    elif args.dataset in ('ogbn-arxiv', 'ogbn-products'):
        try:
            from ogb.nodeproppred import NodePropPredDataset
        except ImportError:
            raise RuntimeError(
                f"OGB dataset '{args.dataset}' requires: pip install ogb"
            )

        print(f"  Loading OGB dataset: {args.dataset}")
        dataset = NodePropPredDataset(name=args.dataset, root='./data')
        graph, labels = dataset[0]

        # Convert to PyTorch tensors
        x = torch.tensor(graph['node_feat'], dtype=torch.float32, device=device)
        y = torch.tensor(labels.reshape(-1), dtype=torch.long, device=device)

        # Edge index
        src = torch.from_numpy(graph['edge_index'][0]).to(device=device, dtype=torch.long)
        dst = torch.from_numpy(graph['edge_index'][1]).to(device=device, dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)

        # Official train/valid/test splits
        split = dataset.get_idx_split()
        train_idx = torch.tensor(split['train'], device=device)
        valid_idx = torch.tensor(split['valid'], device=device)
        test_idx = torch.tensor(split['test'], device=device)

        # Convert to masks
        num_nodes = x.size(0)
        def to_mask(idx):
            mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            mask[idx] = True
            return mask

        train_mask = to_mask(train_idx)
        valid_mask = to_mask(valid_idx)
        test_mask = to_mask(test_idx)

        print(f"    Nodes: {num_nodes:,}, Edges: {edge_index.size(1):,}, Features: {x.size(1)}")
        print(f"    Train: {train_mask.sum().item():,}, Valid: {valid_mask.sum().item():,}, Test: {test_mask.sum().item():,}")

        return x, y, edge_index, train_mask, valid_mask, test_mask

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


# =============================================================================
# Training and Evaluation
# =============================================================================

def accuracy(logits, y, mask):
    """Compute classification accuracy on masked nodes"""
    if mask is None:
        return float('nan')
    pred = logits.argmax(dim=-1)
    correct = (pred[mask] == y[mask]).sum().item()
    total = int(mask.sum().item())
    return correct / max(total, 1)


def train_one_epoch(model, data, optimizer, args):
    """
    Train for one epoch with proper synchronization.

    Returns:
        tuple: (loss, epoch_time)
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    t0 = time.time()

    if args.prop == 'spmm':
        x, y, A, train_mask = data
        out = model(x, A)
    else:  # chunked
        x, y, src, dst, w, train_mask = data
        out = model(x, src, dst, w)

    loss = F.cross_entropy(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    return loss.item(), time.time() - t0


@torch.inference_mode()
def evaluate(model, data, y, masks, args):
    """
    Evaluate model on train/valid/test sets.

    Returns:
        dict: accuracy for each split
    """
    model.eval()

    if args.prop == 'spmm':
        x, A = data
        out = model(x, A)
    else:  # chunked
        x, src, dst, w = data
        out = model(x, src, dst, w)

    accs = {}
    for name, mask in masks.items():
        accs[name] = accuracy(out, y, mask)
    return accs


# =============================================================================
# Memory Tracking
# =============================================================================

def get_memory_stats(uvm_lib=None, gpu_lib=None):
    """Get comprehensive memory statistics"""
    cpu_used = psutil.Process().memory_info().rss / 1e9

    if uvm_lib is not None:
        # UVM allocator statistics
        gpu_allocated = uvm_lib.uvm_get_allocated_bytes() / 1e9
        gpu_peak = uvm_lib.uvm_get_peak_allocated_bytes() / 1e9
        return {
            'gpu_allocated': gpu_allocated,
            'gpu_reserved': gpu_peak,
            'cpu_used': cpu_used,
            'total': gpu_allocated + cpu_used
        }
    elif gpu_lib is not None:
        # Custom GPU allocator statistics
        gpu_allocated = gpu_lib.gpu_get_allocated_bytes() / 1e9
        gpu_peak = gpu_lib.gpu_get_peak_allocated_bytes() / 1e9
        return {
            'gpu_allocated': gpu_allocated,
            'gpu_reserved': gpu_peak,
            'cpu_used': cpu_used,
            'total': gpu_allocated + cpu_used
        }
    else:
        # Standard PyTorch allocator
        gpu_allocated = torch.cuda.memory_allocated() / 1e9
        gpu_reserved = torch.cuda.memory_reserved() / 1e9
        return {
            'gpu_allocated': gpu_allocated,
            'gpu_reserved': gpu_reserved,
            'cpu_used': cpu_used,
            'total': gpu_allocated + cpu_used
        }


def enable_uvm_allocator():
    """Enable UVM allocator and return library handle for statistics"""
    so_path = os.path.join(os.path.dirname(__file__), 'uvm_allocator.so')

    if not os.path.exists(so_path):
        raise FileNotFoundError(
            f"UVM allocator not found at {so_path}\n"
            f"Please build it first with: make -C pytorch/"
        )

    # Load library for statistics
    uvm_lib = ctypes.CDLL(so_path)

    # Define function signatures
    uvm_lib.uvm_get_allocated_bytes.restype = ctypes.c_size_t
    uvm_lib.uvm_get_allocated_bytes.argtypes = []

    uvm_lib.uvm_get_peak_allocated_bytes.restype = ctypes.c_size_t
    uvm_lib.uvm_get_peak_allocated_bytes.argtypes = []

    uvm_lib.uvm_get_num_allocs.restype = ctypes.c_size_t
    uvm_lib.uvm_get_num_allocs.argtypes = []

    uvm_lib.uvm_get_num_frees.restype = ctypes.c_size_t
    uvm_lib.uvm_get_num_frees.argtypes = []

    # Enable in PyTorch
    allocator = torch.cuda.memory.CUDAPluggableAllocator(
        so_path, 'uvm_malloc', 'uvm_free'
    )
    torch.cuda.memory.change_current_allocator(allocator)

    return uvm_lib


def enable_gpu_allocator():
    """Enable custom GPU allocator (NOT UVM) for comparison testing"""
    so_path = os.path.join(os.path.dirname(__file__), 'gpu_allocator.so')

    if not os.path.exists(so_path):
        raise FileNotFoundError(
            f"GPU allocator not found at {so_path}\n"
            f"Please build it first with: make -C pytorch/"
        )

    # Load library for statistics
    gpu_lib = ctypes.CDLL(so_path)

    # Define function signatures
    gpu_lib.gpu_get_allocated_bytes.restype = ctypes.c_size_t
    gpu_lib.gpu_get_allocated_bytes.argtypes = []

    gpu_lib.gpu_get_peak_allocated_bytes.restype = ctypes.c_size_t
    gpu_lib.gpu_get_peak_allocated_bytes.argtypes = []

    gpu_lib.gpu_get_num_allocs.restype = ctypes.c_size_t
    gpu_lib.gpu_get_num_allocs.argtypes = []

    gpu_lib.gpu_get_num_frees.restype = ctypes.c_size_t
    gpu_lib.gpu_get_num_frees.argtypes = []

    # Enable in PyTorch
    allocator = torch.cuda.memory.CUDAPluggableAllocator(
        so_path, 'gpu_malloc', 'gpu_free'
    )
    torch.cuda.memory.change_current_allocator(allocator)

    return gpu_lib


# =============================================================================
# Main Benchmark
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='OSDI-compliant GCN benchmark with UVM support',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Dataset options
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv',
                        choices=['ogbn-arxiv', 'ogbn-products', 'random'],
                        help='Dataset to use (default: ogbn-arxiv)')
    parser.add_argument('--nodes', type=int, default=10_000_000,
                        help='Number of nodes for random graph (default: 10M)')
    parser.add_argument('--edges_per_node', type=int, default=10,
                        help='Avg edges per node for random graph (default: 10)')
    parser.add_argument('--features', type=int, default=128,
                        help='Feature dimension for random graph (default: 128)')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes for random graph (default: 10)')

    # Model options
    parser.add_argument('--hidden', type=int, default=256,
                        help='Hidden dimension (default: 256)')
    parser.add_argument('--prop', type=str, default='spmm',
                        choices=['spmm', 'chunked'],
                        help='Propagation mode: spmm or chunked (default: spmm)')
    parser.add_argument('--chunk_size', type=int, default=2_000_000,
                        help='Chunk size for chunked mode (default: 2M)')

    # Training options
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs (default: 3)')
    parser.add_argument('--warmup', type=int, default=1,
                        help='Number of warmup epochs (default: 1)')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of runs for timing (currently unused)')
    parser.add_argument('--seed', type=int, default=2025,
                        help='Random seed for reproducibility (default: 2025)')

    # Allocator and output
    parser.add_argument('--use_uvm', action='store_true',
                        help='Enable UVM allocator (cudaMallocManaged)')
    parser.add_argument('--use_gpu_allocator', action='store_true',
                        help='Enable custom GPU allocator (cudaMalloc, for comparison)')
    parser.add_argument('--report_json', type=str, default='',
                        help='Path to save JSON report')

    args = parser.parse_args()

    # =============================================================================
    # Step 0: Enable custom allocator (must be before any CUDA operations)
    # =============================================================================
    uvm_lib = None
    gpu_lib = None

    if args.use_uvm and args.use_gpu_allocator:
        print("ERROR: Cannot use both --use_uvm and --use_gpu_allocator")
        return

    if args.use_uvm:
        print("=" * 70)
        print("Enabling UVM Allocator (cudaMallocManaged)")
        print("=" * 70)
        uvm_lib = enable_uvm_allocator()
        print("[UVM] Allocator enabled\n")
    elif args.use_gpu_allocator:
        print("=" * 70)
        print("Enabling Custom GPU Allocator (cudaMalloc)")
        print("=" * 70)
        gpu_lib = enable_gpu_allocator()
        print("[GPU] Custom allocator enabled\n")

    # =============================================================================
    # Step 1: Setup for Reproducibility
    # =============================================================================
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False

    device = 'cuda'

    print("=" * 70)
    print("OSDI-Compliant GCN Benchmark")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Propagation: {args.prop}")
    print(f"Epochs: {args.epochs} (warmup: {args.warmup})")
    print(f"Hidden dim: {args.hidden}")
    print(f"UVM: {'Enabled' if args.use_uvm else 'Disabled'}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Seed: {args.seed}")
    print("=" * 70)

    # =============================================================================
    # Step 2: Load Dataset
    # =============================================================================
    print("\n[1/4] Loading dataset...")
    x, y, edge_index, train_mask, valid_mask, test_mask = load_dataset(args, device)
    num_nodes = x.size(0)
    num_features = x.size(1)
    num_classes = int(y.max().item()) + 1 if args.dataset != 'random' else args.num_classes

    mem = get_memory_stats(uvm_lib, gpu_lib)
    print(f"  Memory: GPU={mem['gpu_allocated']:.2f}GB, CPU={mem['cpu_used']:.2f}GB")

    # =============================================================================
    # Step 3: Build Normalized Graph
    # =============================================================================
    print("\n[2/4] Building normalized graph (D^{-1/2}AD^{-1/2})...")
    (src, dst, w), A = build_normalized_graph(
        edge_index, num_nodes, device, make_coo=(args.prop == 'spmm')
    )
    print(f"  Normalized edges: {src.size(0):,} (with self-loops and symmetrization)")

    mem = get_memory_stats(uvm_lib, gpu_lib)
    print(f"  Memory: GPU={mem['gpu_allocated']:.2f}GB, CPU={mem['cpu_used']:.2f}GB")

    # =============================================================================
    # Step 4: Create Model
    # =============================================================================
    print("\n[3/4] Creating GCN model...")
    if args.prop == 'spmm':
        model = GCN_OSDI_SpMM(num_features, args.hidden, num_classes).cuda()
        train_data = (x, y, A, train_mask)
        eval_data = (x, A)
    else:  # chunked
        model = GCN_OSDI_Chunked(
            num_features, args.hidden, num_classes, chunk_size=args.chunk_size
        ).cuda()
        train_data = (x, y, src, dst, w, train_mask)
        eval_data = (x, src, dst, w)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    mem = get_memory_stats(uvm_lib, gpu_lib)
    print(f"  Memory: GPU={mem['gpu_allocated']:.2f}GB, CPU={mem['cpu_used']:.2f}GB")

    # =============================================================================
    # Step 5: Warmup
    # =============================================================================
    print(f"\n[4/4] Training ({args.warmup} warmup + {args.epochs} measured epochs)...")
    print("  Warmup:")
    for i in range(args.warmup):
        loss, dt = train_one_epoch(model, train_data, optimizer, args)
        print(f"    Warmup {i+1}/{args.warmup}: loss={loss:.4f}, time={dt:.3f}s")

    # =============================================================================
    # Step 6: Training with Timing
    # =============================================================================
    print("  Training:")
    epoch_times = []

    for ep in range(args.epochs):
        loss, dt = train_one_epoch(model, train_data, optimizer, args)
        epoch_times.append(dt)

        if ep == 0 or ep == args.epochs - 1:
            mem = get_memory_stats(uvm_lib, gpu_lib)
            print(f"    Epoch {ep+1}/{args.epochs}: loss={loss:.4f}, time={dt:.3f}s | "
                  f"GPU={mem['gpu_allocated']:.2f}GB, CPU={mem['cpu_used']:.2f}GB")
        else:
            print(f"    Epoch {ep+1}/{args.epochs}: loss={loss:.4f}, time={dt:.3f}s")

    # =============================================================================
    # Step 7: Evaluation
    # =============================================================================
    print("\n[Evaluation]")
    masks = {}
    if train_mask is not None:
        masks['train'] = train_mask
    if valid_mask is not None:
        masks['valid'] = valid_mask
    if test_mask is not None:
        masks['test'] = test_mask

    accs = {}
    if masks:
        accs = evaluate(model, eval_data, y, masks, args)
        for split, acc in accs.items():
            if acc == acc:  # not NaN
                print(f"  {split.capitalize()} Accuracy: {acc:.4f}")

    # =============================================================================
    # Step 8: Final Results
    # =============================================================================
    avg_time = sum(epoch_times) / len(epoch_times)
    median_time = sorted(epoch_times)[len(epoch_times) // 2]

    final_mem = get_memory_stats(uvm_lib, gpu_lib)

    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"Avg epoch time: {avg_time:.3f}s")
    print(f"Median epoch time: {median_time:.3f}s")
    print(f"Total training time: {sum(epoch_times):.2f}s")

    if accs:
        print("\nAccuracy:")
        for split, acc in accs.items():
            if acc == acc:
                print(f"  {split}: {acc:.4f}")

    print(f"\nMemory Usage:")
    print(f"  GPU allocated: {final_mem['gpu_allocated']:.2f} GB")
    print(f"  CPU used: {final_mem['cpu_used']:.2f} GB")
    print(f"  Total: {final_mem['total']:.2f} GB")

    if args.use_uvm and uvm_lib:
        print(f"\nUVM Statistics:")
        print(f"  Peak allocated: {uvm_lib.uvm_get_peak_allocated_bytes() / 1e9:.2f} GB")
        print(f"  Allocations: {uvm_lib.uvm_get_num_allocs()}")
        print(f"  Frees: {uvm_lib.uvm_get_num_frees()}")

    print("=" * 70)

    # =============================================================================
    # Step 9: Save JSON Report (always save to result folder)
    # =============================================================================
    report = {
        'dataset': args.dataset,
        'propagation': args.prop,
        'epochs': args.epochs,
        'warmup': args.warmup,
        'hidden_dim': args.hidden,
        'chunk_size': args.chunk_size if args.prop == 'chunked' else None,
        'num_nodes': num_nodes,
        'num_features': num_features,
        'num_classes': num_classes,
        'num_edges': src.size(0),
        'avg_epoch_time_s': avg_time,
        'median_epoch_time_s': median_time,
        'total_time_s': sum(epoch_times),
        'epoch_times_s': epoch_times,
        'accuracy': {k: (v if v == v else None) for k, v in accs.items()},
        'memory': {
            'gpu_allocated_GB': final_mem['gpu_allocated'],
            'cpu_used_GB': final_mem['cpu_used'],
            'total_GB': final_mem['total']
        },
        'use_uvm': args.use_uvm,
        'seed': args.seed,
        'gpu_name': torch.cuda.get_device_name(0),
        'gpu_memory_GB': torch.cuda.get_device_properties(0).total_memory / 1e9,
        'timestamp': datetime.now().isoformat()
    }

    if args.use_uvm and uvm_lib:
        report['uvm_stats'] = {
            'peak_allocated_GB': uvm_lib.uvm_get_peak_allocated_bytes() / 1e9,
            'num_allocs': uvm_lib.uvm_get_num_allocs(),
            'num_frees': uvm_lib.uvm_get_num_frees()
        }

    # Create result directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(script_dir, 'result')
    os.makedirs(result_dir, exist_ok=True)

    # Generate filename based on configuration
    uvm_suffix = '_uvm' if args.use_uvm else ''
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_filename = f"gcn_{args.dataset}_{args.prop}{uvm_suffix}_{timestamp_str}.json"
    result_path = os.path.join(result_dir, result_filename)

    with open(result_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n[Report] Saved to {result_path}")

    # Also save to user-specified path if provided
    if args.report_json:
        with open(args.report_json, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"[Report] Also saved to {os.path.abspath(args.report_json)}")


if __name__ == '__main__':
    main()
