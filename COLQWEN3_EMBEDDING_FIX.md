# ColQwen3 Multi-Vector Embedding KV Cache Fix

## Problem Description

When running the SGLang server with ColQwen3 (a multi-vector embedding model), subsequent requests with the same input returned embeddings with incorrect dimensions:

- **First request**: Correct dimensions `[13, 320]` for text, `[486, 320]` for images
- **Subsequent requests**: Incorrect dimensions `[1, 320]` due to KV caching

## Root Cause

ColQwen3 is a multi-vector embedding model that outputs **per-token embeddings** (not pooled single-vector embeddings). When prefix caching is enabled (default in SGLang):

1. First request: All tokens are processed, full embeddings returned
2. Subsequent identical requests: Most tokens are cached, only 1 new token processed
3. The model only generates embeddings for the processed tokens (based on `extend_seq_lens`)
4. Result: Only 1 embedding returned instead of all per-token embeddings

This is incorrect for retrieval models like ColQwen3 that use late-interaction (MaxSim) scoring, which requires embeddings for ALL tokens.

## Solution

Disable prefix caching for multi-vector embedding models by:
1. Identifying these models via a configuration flag
2. Skipping prefix matching during request scheduling
3. Handling `None` last_node gracefully throughout the caching infrastructure

## Files Modified

### 1. `python/sglang/srt/configs/model_config.py`

Added identification of multi-vector embedding models:

```python
# Multi-vector embedding models that require per-token embeddings
multivector_embedding_model_archs = [
    "ColQwen3",
    # Add other ColPali-style models here as needed
]

def is_multivector_embedding_model(model_architectures: List[str]):
    """Check if the model is a multi-vector embedding model."""
    return any(
        arch in model_architectures for arch in multivector_embedding_model_archs
    )
```

Added property to `ModelConfig` class:
```python
self.is_multivector_embedding = is_multivector_embedding_model(
    self.hf_config.architectures
)
```

### 2. `python/sglang/srt/managers/scheduler.py`

Skip prefix caching for multi-vector embedding models:

```python
def _prefetch_kvcache(self, req: Req):
    # Skip prefix caching for multi-vector embedding models (like ColQwen3)
    if self.model_config.is_multivector_embedding:
        return
    # ... rest of method
```

```python
# In get_new_batch_prefill():
if self.model_config.is_multivector_embedding:
    req.init_next_round_input(None)
else:
    req.init_next_round_input(self.tree_cache)
```

### 3. `python/sglang/srt/managers/schedule_policy.py`

Handle `None` last_node in locking mechanisms:

```python
@contextmanager
def _lock_node(self, last_node: TreeNode):
    # If last_node is None, there's nothing to lock
    if last_node is None:
        yield None
        return
    # ... rest of method
```

Also added `if req.last_node is not None:` guards around `inc_lock_ref` calls in `add_one_req()`.

### 4. `python/sglang/srt/mem_cache/radix_cache.py`

Handle `None` node in lock reference methods:

```python
def inc_lock_ref(self, node: TreeNode):
    if self.disable:
        return 0
    # Handle None node (e.g., for multi-vector embedding models)
    if node is None:
        return 0
    # ... rest of method

def dec_lock_ref(self, node: TreeNode):
    if self.disable:
        return 0
    # Handle None node (e.g., for multi-vector embedding models)
    if node is None:
        return 0
    # ... rest of method
```

### 5. `python/sglang/srt/disaggregation/decode.py`

Same fix as scheduler.py for disaggregated inference path:

```python
if self.model_config.is_multivector_embedding:
    req.init_next_round_input(None)
else:
    req.init_next_round_input(self.tree_cache)
```

## Testing

After applying the fix:

1. Start the server:
```bash
SGLANG_DISABLE_CUDNN_CHECK=1 uv run python -m sglang.launch_server \
    --model-path TomoroAI/tomoro-ai-colqwen3-embed-4b-awq \
    --is-embedding --trust-remote-code --port 30000
```

2. Run the comparison script multiple times:
```bash
uv run python compare_colqwen3_embeddings.py --skip-hf
uv run python compare_colqwen3_embeddings.py --skip-hf
```

3. Verify that both runs return embeddings with consistent shapes:
   - Text: `[13, 320]` (or similar based on input)
   - Image: `[486, 320]` (or similar based on image resolution)

## Impact

- **ColQwen3 and similar models**: Now correctly return all per-token embeddings on every request
- **Other models**: No impact - prefix caching continues to work normally
- **Performance**: Multi-vector embedding models will not benefit from prefix caching, but this is necessary for correctness

## Future Work

To add support for other multi-vector embedding models (e.g., ColPali, ColBERT variants), add their architecture names to `multivector_embedding_model_archs` in `model_config.py`.
