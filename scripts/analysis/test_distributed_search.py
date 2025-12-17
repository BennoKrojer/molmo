#!/usr/bin/env python3
"""
Quick test for distributed GPU search without loading full model/embeddings.
Run with: torchrun --nproc_per_node=4 scripts/analysis/test_distributed_search.py
"""

import torch
import torch.distributed as dist
import os
import sys

def init_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()

def distributed_topk_search(query_embeddings, contextual_embeddings_shard, global_offset, top_k, world_size):
    """
    Distributed exact nearest neighbor search across sharded embeddings.
    """
    device = query_embeddings.device
    num_queries = query_embeddings.shape[0]
    
    # Step 1: Local similarity computation
    local_similarity = torch.matmul(query_embeddings, contextual_embeddings_shard.T)
    
    # Step 2: Local top-k
    local_k = min(top_k, local_similarity.shape[-1])
    local_values, local_indices = torch.topk(local_similarity, k=local_k, dim=-1)
    
    # Convert local indices to global indices
    local_indices_global = local_indices + global_offset
    
    # Step 3: All-gather from all GPUs
    gathered_values_list = [torch.zeros_like(local_values) for _ in range(world_size)]
    gathered_indices_list = [torch.zeros_like(local_indices_global) for _ in range(world_size)]
    
    dist.all_gather(gathered_values_list, local_values)
    dist.all_gather(gathered_indices_list, local_indices_global)
    
    # Step 4: Concatenate and find global top-k
    all_values = torch.cat(gathered_values_list, dim=-1)
    all_indices = torch.cat(gathered_indices_list, dim=-1)
    
    # Final top-k across all shards
    global_top_values, merge_indices = torch.topk(all_values, k=top_k, dim=-1)
    global_top_indices = torch.gather(all_indices, dim=-1, index=merge_indices)
    
    return global_top_values, global_top_indices

def test_distributed_search():
    local_rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    if local_rank == 0:
        print(f"\n{'='*60}")
        print(f"Testing Distributed GPU Search with {world_size} GPUs")
        print(f"{'='*60}\n")
    
    # Simulate embeddings (smaller scale for quick test)
    total_embeddings = 100000  # 100k instead of 3.3M
    embedding_dim = 3584
    num_queries = 576  # Like one image
    top_k = 5
    
    # Create full embeddings on CPU (only for ground truth comparison)
    if local_rank == 0:
        print(f"Creating test data: {total_embeddings:,} embeddings, {num_queries} queries")
    
    torch.manual_seed(42)  # Same data on all ranks
    full_embeddings = torch.randn(total_embeddings, embedding_dim)
    full_embeddings = torch.nn.functional.normalize(full_embeddings, dim=-1)
    
    queries = torch.randn(num_queries, embedding_dim)
    queries = torch.nn.functional.normalize(queries, dim=-1)
    
    # Shard embeddings
    shard_size = total_embeddings // world_size
    start_idx = local_rank * shard_size
    end_idx = start_idx + shard_size if local_rank < world_size - 1 else total_embeddings
    
    shard = full_embeddings[start_idx:end_idx].to(device)
    queries_gpu = queries.to(device)
    
    if local_rank == 0:
        print(f"Each GPU has {shard.shape[0]:,} embeddings ({shard.numel() * 4 / 1e9:.3f} GB)")
    
    # Distributed search
    dist.barrier()
    
    import time
    start = time.time()
    
    dist_values, dist_indices = distributed_topk_search(
        queries_gpu, shard, start_idx, top_k, world_size
    )
    
    torch.cuda.synchronize()
    dist_time = time.time() - start
    
    if local_rank == 0:
        print(f"\n⏱️  Distributed search time: {dist_time:.3f}s")
        print(f"   Per query: {dist_time/num_queries*1000:.2f}ms")
    
    # Ground truth: brute force on single GPU
    if local_rank == 0:
        print(f"\nComputing ground truth (single GPU brute force)...")
        full_embeddings_gpu = full_embeddings.to(device)
        
        start = time.time()
        similarity = torch.matmul(queries_gpu, full_embeddings_gpu.T)
        gt_values, gt_indices = torch.topk(similarity, k=top_k, dim=-1)
        torch.cuda.synchronize()
        bf_time = time.time() - start
        
        print(f"⏱️  Brute force time: {bf_time:.3f}s")
        
        # Compare results
        dist_indices_cpu = dist_indices.cpu()
        gt_indices_cpu = gt_indices.cpu()
        
        # Check if indices match (top-1 especially important)
        top1_match = (dist_indices_cpu[:, 0] == gt_indices_cpu[:, 0]).float().mean()
        topk_match = (dist_indices_cpu == gt_indices_cpu).float().mean()
        
        print(f"\n📊 Accuracy:")
        print(f"   Top-1 match: {top1_match*100:.1f}%")
        print(f"   Top-{top_k} exact match: {topk_match*100:.1f}%")
        
        if top1_match < 0.99:
            print(f"\n⚠️  WARNING: Top-1 accuracy is below 99%!")
            print(f"   First 5 distributed: {dist_indices_cpu[:5, 0].tolist()}")
            print(f"   First 5 ground truth: {gt_indices_cpu[:5, 0].tolist()}")
        else:
            print(f"\n✅ SUCCESS: Distributed search matches ground truth!")
        
        # Clean up
        del full_embeddings_gpu
    
    dist.barrier()
    
    if local_rank == 0:
        print(f"\n{'='*60}")
        print(f"Test complete!")
        print(f"{'='*60}\n")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    test_distributed_search()

