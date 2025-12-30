"""
Compare ColQwen3 multimodal embeddings generated via SGLang server vs HuggingFace directly.

Usage:
1. First, start the SGLang server with:
   python -m sglang.launch_server --model TomoroAI/tomoro-colqwen3-embed-4b \
       --trust-remote-code --port 30000 --is-embedding

2. Then run this script:
   python compare_colqwen3_embeddings.py
"""

import argparse
import time
from io import BytesIO
from typing import List, Tuple

import numpy as np
import requests
import torch
from PIL import Image, UnidentifiedImageError

# Configuration
MODEL_ID = "TomoroAI/tomoro-colqwen3-embed-4b"
SGLANG_URL = "http://127.0.0.1:30000"
DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Test data - using an image with a text query
# Use a more accessible image URL (picsum provides test images without auth)
TEST_IMAGE_URL = "https://picsum.photos/id/1/800/600"
TEST_TEXT_QUERY = "Describe the image"


def load_image(url: str) -> Image.Image:
    """Load an image from a URL with fallback headers for CDN access."""
    for headers in ({}, {"User-Agent": "Mozilla/5.0 (compatible; ColQwen3-demo/1.0)"}):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 403:
                continue
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")
        except UnidentifiedImageError as e:
            raise RuntimeError(f"Failed to decode image from {url}") from e
    raise RuntimeError(f"Could not fetch image (HTTP 403) from {url}")


# =============================================================================
# HuggingFace Direct Method
# =============================================================================

def get_hf_model_and_processor():
    """Load HuggingFace model and processor."""
    from transformers import AutoModel, AutoProcessor

    print("Loading HuggingFace model and processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        max_num_visual_tokens=1280,
    )
    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        device_map=DEVICE,
    ).eval()
    print("HuggingFace model loaded successfully.")
    return model, processor


def encode_text_hf(model, processor, texts: List[str]) -> List[torch.Tensor]:
    """Encode text queries using HuggingFace model."""
    batch = processor.process_texts(texts=texts)
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    with torch.inference_mode():
        out = model(**batch)
        embeddings = out.embeddings.to(torch.float32).cpu()
    return [embeddings[i] for i in range(embeddings.shape[0])]


def encode_image_hf(model, processor, images: List[Image.Image]) -> List[torch.Tensor]:
    """Encode images using HuggingFace model."""
    features = processor.process_images(images=images)
    features = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in features.items()}
    with torch.inference_mode():
        out = model(**features)
        embeddings = out.embeddings.to(torch.float32).cpu()
    return [embeddings[i] for i in range(embeddings.shape[0])]


# =============================================================================
# SGLang Server Method
# =============================================================================

def check_sglang_server(url: str) -> bool:
    """Check if SGLang server is running."""
    try:
        resp = requests.get(f"{url}/health", timeout=5)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def encode_text_sglang(url: str, texts: List[str]) -> List[List[List[float]]]:
    """Encode text queries using SGLang server."""
    # For text-only, use simple string format (not multimodal format)
    payload = {
        "model": MODEL_ID,
        "input": texts[0] if len(texts) == 1 else texts,
    }
    response = requests.post(f"{url}/v1/embeddings", json=payload, timeout=120)
    if response.status_code != 200:
        print(f"  Error response: {response.text}")
    response.raise_for_status()
    result = response.json()

    # Extract embeddings from response
    embeddings = []
    for item in result.get("data", []):
        emb = item.get("embedding")
        embeddings.append(emb)
    return embeddings


def encode_image_sglang(url: str, image_urls: List[str]) -> List[List[List[float]]]:
    """Encode images using SGLang server."""
    # Use multimodal format with text prompt and image
    # ColQwen3 requires a text prompt for image encoding (used as query context)
    payload = {
        "model": MODEL_ID,
        "input": [{"text": "Describe the image.", "image": img_url} for img_url in image_urls],
    }
    response = requests.post(f"{url}/v1/embeddings", json=payload, timeout=120)
    if response.status_code != 200:
        print(f"  Error response: {response.text}")
    response.raise_for_status()
    result = response.json()

    # Extract embeddings from response
    embeddings = []
    for item in result.get("data", []):
        emb = item.get("embedding")
        embeddings.append(emb)
    return embeddings


# =============================================================================
# Comparison Utilities
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def compare_embeddings(
    emb_hf: torch.Tensor,
    emb_sglang: List[List[float]],
    name: str = "embedding"
) -> dict:
    """Compare embeddings from HuggingFace and SGLang."""
    # Convert to numpy for comparison
    if isinstance(emb_hf, torch.Tensor):
        hf_np = emb_hf.numpy()
    else:
        hf_np = np.array(emb_hf)

    sglang_np = np.array(emb_sglang)

    print(f"\n{'='*60}")
    print(f"Comparison for: {name}")
    print(f"{'='*60}")

    print(f"\nHuggingFace embedding shape: {hf_np.shape}")
    print(f"SGLang embedding shape: {sglang_np.shape}")

    # Check if shapes match
    if hf_np.shape != sglang_np.shape:
        print(f"\n⚠ Shape mismatch! Shapes are different.")
        # Try to compare what we can
        min_tokens = min(hf_np.shape[0], sglang_np.shape[0])
        print(f"Comparing first {min_tokens} tokens...")
        hf_np = hf_np[:min_tokens]
        sglang_np = sglang_np[:min_tokens]

    # Compute statistics
    diff = hf_np - sglang_np
    mse = float(np.mean(diff ** 2))
    max_abs_diff = float(np.max(np.abs(diff)))
    mean_abs_diff = float(np.mean(np.abs(diff)))

    # Compute per-token cosine similarities
    cosine_sims = []
    for i in range(hf_np.shape[0]):
        sim = cosine_similarity(hf_np[i], sglang_np[i])
        cosine_sims.append(sim)

    mean_cosine = np.mean(cosine_sims)
    min_cosine = np.min(cosine_sims)
    max_cosine = np.max(cosine_sims)

    print(f"\nStatistics:")
    print(f"  Mean Squared Error (MSE): {mse:.6e}")
    print(f"  Max Absolute Difference:  {max_abs_diff:.6e}")
    print(f"  Mean Absolute Difference: {mean_abs_diff:.6e}")
    print(f"\nPer-token Cosine Similarity:")
    print(f"  Mean:    {mean_cosine:.6f}")
    print(f"  Min:     {min_cosine:.6f}")
    print(f"  Max:     {max_cosine:.6f}")

    # Sample comparison of first few values
    print(f"\nSample values (first 5 values of first token):")
    print(f"  HuggingFace: {hf_np[0, :5].tolist()}")
    print(f"  SGLang:      {sglang_np[0, :5].tolist()}")

    return {
        "mse": mse,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "mean_cosine_sim": mean_cosine,
        "min_cosine_sim": min_cosine,
        "max_cosine_sim": max_cosine,
        "hf_shape": hf_np.shape,
        "sglang_shape": sglang_np.shape,
    }


def maxsim_score(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
    """Compute MaxSim score between query and document embeddings."""
    # query_emb: [Q, D], doc_emb: [K, D]
    # MaxSim: for each query token, find max similarity across all doc tokens
    # Then sum or average across query tokens
    sim_matrix = query_emb @ doc_emb.T  # [Q, K]
    max_sims = np.max(sim_matrix, axis=1)  # [Q]
    return float(np.sum(max_sims))


def main():
    parser = argparse.ArgumentParser(description="Compare ColQwen3 embeddings")
    parser.add_argument("--sglang-url", default=SGLANG_URL, help="SGLang server URL")
    parser.add_argument("--image-url", default=TEST_IMAGE_URL, help="Image URL to encode")
    parser.add_argument("--text", default=TEST_TEXT_QUERY, help="Text query to encode")
    parser.add_argument("--skip-hf", action="store_true", help="Skip HuggingFace comparison")
    args = parser.parse_args()

    print("=" * 70)
    print("ColQwen3 Multimodal Embedding Comparison")
    print("=" * 70)
    print(f"\nModel: {MODEL_ID}")
    print(f"Image URL: {args.image_url}")
    print(f"Text query: {args.text}")

    # Check SGLang server
    print(f"\nChecking SGLang server at {args.sglang_url}...")
    if not check_sglang_server(args.sglang_url):
        print("⚠ SGLang server is not running!")
        print("\nPlease start the server with:")
        print(f"  python -m sglang.launch_server --model {MODEL_ID} \\")
        print("      --trust-remote-code --port 30000 --is-embedding")
        return
    print("✓ SGLang server is running")

    # =========================================================================
    # SGLang embeddings
    # =========================================================================
    print("\n" + "-" * 70)
    print("Generating embeddings via SGLang server...")
    print("-" * 70)

    # Text embedding via SGLang
    print("\n[SGLang] Encoding text query...")
    t0 = time.time()
    sglang_text_embs = encode_text_sglang(args.sglang_url, [args.text])
    t1 = time.time()
    print(f"  ✓ Text embedding generated in {t1-t0:.2f}s")
    if sglang_text_embs and sglang_text_embs[0]:
        sglang_text_emb = sglang_text_embs[0]
        print(f"  Shape: [{len(sglang_text_emb)}, {len(sglang_text_emb[0]) if sglang_text_emb else 0}]")
    else:
        print("  ⚠ No text embedding returned")
        sglang_text_emb = None

    # Image embedding via SGLang
    print("\n[SGLang] Encoding image...")
    t0 = time.time()
    sglang_img_embs = encode_image_sglang(args.sglang_url, [args.image_url])
    t1 = time.time()
    print(f"  ✓ Image embedding generated in {t1-t0:.2f}s")
    if sglang_img_embs and sglang_img_embs[0]:
        sglang_img_emb = sglang_img_embs[0]
        print(f"  Shape: [{len(sglang_img_emb)}, {len(sglang_img_emb[0]) if sglang_img_emb else 0}]")
    else:
        print("  ⚠ No image embedding returned")
        sglang_img_emb = None

    if args.skip_hf:
        print("\nSkipping HuggingFace comparison (--skip-hf flag set)")

        # Just compute MaxSim between SGLang embeddings
        if sglang_text_emb and sglang_img_emb:
            query_np = np.array(sglang_text_emb)
            doc_np = np.array(sglang_img_emb)
            score = maxsim_score(query_np, doc_np)
            print(f"\n[SGLang] MaxSim score (text -> image): {score:.4f}")
        return

    # =========================================================================
    # HuggingFace embeddings
    # =========================================================================
    print("\n" + "-" * 70)
    print("Generating embeddings via HuggingFace directly...")
    print("-" * 70)

    model, processor = get_hf_model_and_processor()

    # Load image
    print("\nLoading image...")
    image = load_image(args.image_url)
    print(f"  ✓ Image loaded: {image.size}")

    # Text embedding via HuggingFace
    print("\n[HuggingFace] Encoding text query...")
    t0 = time.time()
    hf_text_embs = encode_text_hf(model, processor, [args.text])
    t1 = time.time()
    print(f"  ✓ Text embedding generated in {t1-t0:.2f}s")
    hf_text_emb = hf_text_embs[0]
    print(f"  Shape: {hf_text_emb.shape}")

    # Image embedding via HuggingFace
    print("\n[HuggingFace] Encoding image...")
    t0 = time.time()
    hf_img_embs = encode_image_hf(model, processor, [image])
    t1 = time.time()
    print(f"  ✓ Image embedding generated in {t1-t0:.2f}s")
    hf_img_emb = hf_img_embs[0]
    print(f"  Shape: {hf_img_emb.shape}")

    # =========================================================================
    # Compare embeddings
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARING EMBEDDINGS")
    print("=" * 70)

    # Compare text embeddings
    if sglang_text_emb:
        text_stats = compare_embeddings(hf_text_emb, sglang_text_emb, "Text Query Embedding")
    else:
        print("\n⚠ Cannot compare text embeddings - SGLang returned None")
        text_stats = None

    # Compare image embeddings
    if sglang_img_emb:
        img_stats = compare_embeddings(hf_img_emb, sglang_img_emb, "Image Embedding")
    else:
        print("\n⚠ Cannot compare image embeddings - SGLang returned None")
        img_stats = None

    # =========================================================================
    # MaxSim retrieval comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("MaxSim RETRIEVAL COMPARISON")
    print("=" * 70)

    # Compute MaxSim scores
    hf_query_np = hf_text_emb.numpy()
    hf_doc_np = hf_img_emb.numpy()
    hf_score = maxsim_score(hf_query_np, hf_doc_np)
    print(f"\n[HuggingFace] MaxSim score (text -> image): {hf_score:.4f}")

    if sglang_text_emb and sglang_img_emb:
        sglang_query_np = np.array(sglang_text_emb)
        sglang_doc_np = np.array(sglang_img_emb)
        sglang_score = maxsim_score(sglang_query_np, sglang_doc_np)
        print(f"[SGLang]      MaxSim score (text -> image): {sglang_score:.4f}")
        print(f"\nScore difference: {abs(hf_score - sglang_score):.4f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if text_stats:
        print(f"\nText Embedding:")
        print(f"  - Mean cosine similarity: {text_stats['mean_cosine_sim']:.6f}")
        if text_stats['mean_cosine_sim'] > 0.99:
            print("  - ✓ Excellent match!")
        elif text_stats['mean_cosine_sim'] > 0.95:
            print("  - ✓ Good match")
        else:
            print("  - ⚠ Significant differences detected")

    if img_stats:
        print(f"\nImage Embedding:")
        print(f"  - Mean cosine similarity: {img_stats['mean_cosine_sim']:.6f}")
        if img_stats['mean_cosine_sim'] > 0.99:
            print("  - ✓ Excellent match!")
        elif img_stats['mean_cosine_sim'] > 0.95:
            print("  - ✓ Good match")
        else:
            print("  - ⚠ Significant differences detected")


if __name__ == "__main__":
    main()
