# Plan: Enable KV Cache with Multi-Vector Embedding Cache for ColQwen3

## Problem Summary

Multi-vector embedding models like ColQwen3 output **per-token embeddings** (not pooled). The current fix disables prefix caching entirely because:
- With prefix caching, only `extend_seq_lens` new tokens are processed
- Only new token embeddings are returned, losing prefix token embeddings
- This sacrifices KV cache performance benefits for correctness

## Solution: Embedding Cache

Re-enable KV cache for attention efficiency while adding a separate **Embedding Cache** that:
1. Stores per-token embeddings for prefix sequences
2. On cache hit: retrieves cached embeddings + computes new token embeddings + concatenates
3. Coordinates eviction with KV cache

## Implementation Steps

### Step 1: Create MultivectorEmbeddingCache Class

**New file:** `python/sglang/srt/mem_cache/multivector_embedding_cache.py`

```python
@dataclass
class MultivectorEmbeddingEntry:
    embeddings: torch.Tensor  # [seq_len, embed_dim]
    seq_len: int

class MultivectorEmbeddingCache:
    def __init__(self, max_size_bytes: int)
    def get(self, token_ids: List[int], extra_key: str) -> Optional[torch.Tensor]
    def set(self, token_ids: List[int], embeddings: torch.Tensor, extra_key: str)
    def evict_by_key(self, key: int)
    def _compute_key(self, token_ids, extra_key) -> int  # hash-based
```

- LRU eviction policy (similar to `MultiModalStaticCache`)
- Configurable max size via environment variable

### Step 2: Add Fields to Req Class

**File:** `python/sglang/srt/managers/schedule_batch.py` (around line 619)

Add to `Req.__init__`:
```python
# Multi-vector embedding cache support
self.cached_multivector_embeddings: Optional[torch.Tensor] = None
self.cached_embedding_len: int = 0
```

### Step 3: Add Fields to ForwardBatch

**File:** `python/sglang/srt/model_executor/forward_batch_info.py`

Add to `ForwardBatch`:
```python
cached_multivector_embeddings: Optional[List[torch.Tensor]] = None
cached_embedding_lens: Optional[List[int]] = None
```

### Step 4: Initialize Embedding Cache in Scheduler

**File:** `python/sglang/srt/managers/scheduler.py`

In `init_cache_with_memory_pool()` (around line 720):
```python
if self.model_config.is_multivector_embedding:
    from sglang.srt.mem_cache.multivector_embedding_cache import MultivectorEmbeddingCache
    cache_size = int(os.environ.get("SGLANG_MULTIVECTOR_EMBEDDING_CACHE_MB", 512)) * 1024 * 1024
    self.multivector_embedding_cache = MultivectorEmbeddingCache(cache_size)
else:
    self.multivector_embedding_cache = None
```

### Step 5: Modify Scheduler to Use Both Caches

**File:** `python/sglang/srt/managers/scheduler.py`

#### 5a. Modify `_prefetch_kvcache()` (line ~1598)

Remove the early return for multi-vector models. Instead:
```python
def _prefetch_kvcache(self, req: Req):
    # For multi-vector embedding: use KV cache AND embedding cache
    if self.model_config.is_multivector_embedding:
        # Look up cached embeddings BEFORE prefix matching
        if self.multivector_embedding_cache:
            prefix_token_ids = req.fill_ids  # Will match longest prefix
            cached_emb = self.multivector_embedding_cache.get(
                prefix_token_ids, req.extra_key
            )
            if cached_emb is not None:
                req.cached_multivector_embeddings = cached_emb
                req.cached_embedding_len = cached_emb.shape[0]
    # Continue with normal KV cache prefetching...
```

#### 5b. Modify `get_new_batch_prefill()` (line ~1980)

Remove the special case that passes `None` to `init_next_round_input`:
```python
# Both multi-vector and regular models use tree_cache now
req.init_next_round_input(self.tree_cache)
```

### Step 6: Modify ColQwen3 Forward to Concatenate Embeddings

**File:** `python/sglang/srt/models/colqwen3.py` (line ~324)

Replace the embedding splitting logic:
```python
# Split batch into per-request embeddings
if forward_batch.extend_seq_lens is not None:
    seq_lens = forward_batch.extend_seq_lens.tolist()
    embeddings_list = []

    # Get cached embeddings info
    cached_embs = forward_batch.cached_multivector_embeddings or [None] * len(seq_lens)
    cached_lens = forward_batch.cached_embedding_lens or [0] * len(seq_lens)

    start_idx = 0
    for i, seq_len in enumerate(seq_lens):
        new_emb = embeddings[start_idx : start_idx + seq_len]

        # Concatenate cached (prefix) + new (extend) embeddings
        if cached_embs[i] is not None and cached_lens[i] > 0:
            full_emb = torch.cat([cached_embs[i].to(new_emb.device), new_emb], dim=0)
        else:
            full_emb = new_emb

        embeddings_list.append(full_emb)
        start_idx += seq_len

    return EmbeddingPoolerOutput(embeddings=embeddings_list)
```

### Step 7: Cache Embeddings After Forward

**File:** `python/sglang/srt/managers/scheduler_output_processor_mixin.py` (line ~240)

After extracting embeddings, cache them:
```python
# Cache embeddings for multi-vector models
if (self.model_config.is_multivector_embedding and
    self.multivector_embedding_cache is not None):
    for i, req in enumerate(batch.reqs):
        if not req.is_retracted:
            emb = embeddings[i]
            if isinstance(emb, torch.Tensor):
                self.multivector_embedding_cache.set(
                    req.fill_ids, emb.cpu(), req.extra_key
                )
```

### Step 8: Pass Cached Embeddings to ForwardBatch

**File:** `python/sglang/srt/managers/schedule_batch.py`

In `prepare_for_extend()` (around line 1398), populate ForwardBatch fields:
```python
# Collect cached embeddings for multi-vector models
cached_multivector_embeddings = [r.cached_multivector_embeddings for r in reqs]
cached_embedding_lens = [r.cached_embedding_len for r in reqs]
```

### Step 9: Handle None last_node (Keep Existing Guards)

The existing guards in `radix_cache.py` and `schedule_policy.py` for `None` nodes should be kept as they handle edge cases.

## Files to Modify

| File | Changes |
|------|---------|
| `mem_cache/multivector_embedding_cache.py` | NEW: Embedding cache class |
| `managers/scheduler.py` | Initialize cache, modify `_prefetch_kvcache`, `get_new_batch_prefill` |
| `managers/schedule_batch.py` | Add `Req` fields, pass to `ForwardBatch` |
| `model_executor/forward_batch_info.py` | Add `ForwardBatch` fields |
| `models/colqwen3.py` | Concatenate cached + new embeddings |
| `managers/scheduler_output_processor_mixin.py` | Cache embeddings after forward |

## Testing

1. Start server with ColQwen3
2. Send same document multiple times - verify consistent embedding shapes
3. Send same document with different query suffixes - verify prefix reuse
4. Check memory usage stays within configured limit
5. Compare embeddings with non-cached version for correctness

## Configuration

- `SGLANG_MULTIVECTOR_EMBEDDING_CACHE_MB`: Cache size in MB (default: 512)
