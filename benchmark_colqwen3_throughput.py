"""
Benchmark throughput of ColQwen3 embeddings: SGLang server vs HuggingFace direct.

Usage:
1. First, start the SGLang server with:
   python -m sglang.launch_server --model TomoroAI/tomoro-colqwen3-embed-4b \
       --trust-remote-code --port 30000 --is-embedding

2. Then run this script:
   python benchmark_colqwen3_throughput.py --num-requests 100 --concurrency 8
"""

import argparse
import concurrent.futures
import statistics
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import List, Optional

import numpy as np
import requests
import torch
from PIL import Image

# Configuration
MODEL_ID = "TomoroAI/tomoro-colqwen3-embed-4b"
SGLANG_URL = "http://127.0.0.1:30000"
DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Test data
TEST_IMAGE_URLS = [
    "https://picsum.photos/id/1/800/600",
    "https://picsum.photos/id/10/800/600",
    "https://picsum.photos/id/20/800/600",
    "https://picsum.photos/id/30/800/600",
    "https://picsum.photos/id/40/800/600",
]
TEST_TEXT_QUERIES = [
    "Describe the image",
    "What objects are in this photo?",
    "Summarize the visual content",
    "What is the main subject?",
    "Describe the colors and composition",
]


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    mode: str  # "text" or "image"
    num_requests: int
    total_time: float
    latencies: List[float] = field(default_factory=list)
    errors: int = 0

    @property
    def successful_requests(self) -> int:
        return self.num_requests - self.errors

    @property
    def throughput(self) -> float:
        """Requests per second."""
        if self.total_time == 0:
            return 0
        return self.successful_requests / self.total_time

    @property
    def avg_latency(self) -> float:
        return statistics.mean(self.latencies) if self.latencies else 0

    @property
    def p50_latency(self) -> float:
        return statistics.median(self.latencies) if self.latencies else 0

    @property
    def p95_latency(self) -> float:
        if not self.latencies:
            return 0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def p99_latency(self) -> float:
        if not self.latencies:
            return 0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def min_latency(self) -> float:
        return min(self.latencies) if self.latencies else 0

    @property
    def max_latency(self) -> float:
        return max(self.latencies) if self.latencies else 0


def load_image(url: str) -> Image.Image:
    """Load an image from a URL."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


def preload_images(urls: List[str]) -> List[Image.Image]:
    """Preload images to avoid network latency during benchmark."""
    print("Preloading images...")
    images = []
    for url in urls:
        try:
            images.append(load_image(url))
        except Exception as e:
            print(f"  Warning: Failed to load {url}: {e}")
    print(f"  Loaded {len(images)} images")
    return images


# =============================================================================
# HuggingFace Benchmark
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
    print("HuggingFace model loaded.")
    return model, processor


def benchmark_hf_text(
    model,
    processor,
    texts: List[str],
    num_requests: int,
    batch_size: int = 1,
) -> BenchmarkResult:
    """Benchmark HuggingFace text encoding."""
    latencies = []
    errors = 0

    # Warmup
    print("  Warming up HuggingFace text encoding...")
    batch = processor.process_texts(texts=[texts[0]])
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    with torch.inference_mode():
        _ = model(**batch)
    torch.cuda.synchronize()

    print(f"  Running {num_requests} text encoding requests (batch_size={batch_size})...")
    start_time = time.perf_counter()

    for i in range(num_requests):
        text = texts[i % len(texts)]
        try:
            t0 = time.perf_counter()
            batch = processor.process_texts(texts=[text] * batch_size)
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.inference_mode():
                _ = model(**batch)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append(t1 - t0)
        except Exception as e:
            errors += 1
            print(f"    Error on request {i}: {e}")

    total_time = time.perf_counter() - start_time

    return BenchmarkResult(
        name="HuggingFace",
        mode="text",
        num_requests=num_requests,
        total_time=total_time,
        latencies=latencies,
        errors=errors,
    )


def benchmark_hf_image(
    model,
    processor,
    images: List[Image.Image],
    num_requests: int,
    batch_size: int = 1,
) -> BenchmarkResult:
    """Benchmark HuggingFace image encoding."""
    latencies = []
    errors = 0

    # Warmup
    print("  Warming up HuggingFace image encoding...")
    features = processor.process_images(images=[images[0]])
    features = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in features.items()}
    with torch.inference_mode():
        _ = model(**features)
    torch.cuda.synchronize()

    print(f"  Running {num_requests} image encoding requests (batch_size={batch_size})...")
    start_time = time.perf_counter()

    for i in range(num_requests):
        img = images[i % len(images)]
        try:
            t0 = time.perf_counter()
            features = processor.process_images(images=[img] * batch_size)
            features = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in features.items()}
            with torch.inference_mode():
                _ = model(**features)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append(t1 - t0)
        except Exception as e:
            errors += 1
            print(f"    Error on request {i}: {e}")

    total_time = time.perf_counter() - start_time

    return BenchmarkResult(
        name="HuggingFace",
        mode="image",
        num_requests=num_requests,
        total_time=total_time,
        latencies=latencies,
        errors=errors,
    )


# =============================================================================
# SGLang Benchmark
# =============================================================================

def check_sglang_server(url: str) -> bool:
    """Check if SGLang server is running."""
    try:
        resp = requests.get(f"{url}/health", timeout=5)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def sglang_encode_text_single(url: str, text: str) -> float:
    """Encode a single text and return latency."""
    t0 = time.perf_counter()
    payload = {"model": MODEL_ID, "input": text}
    resp = requests.post(f"{url}/v1/embeddings", json=payload, timeout=120)
    resp.raise_for_status()
    t1 = time.perf_counter()
    return t1 - t0


def sglang_encode_image_single(url: str, image_url: str) -> float:
    """Encode a single image and return latency."""
    t0 = time.perf_counter()
    payload = {
        "model": MODEL_ID,
        "input": [{"text": "Describe the image.", "image": image_url}],
    }
    resp = requests.post(f"{url}/v1/embeddings", json=payload, timeout=120)
    resp.raise_for_status()
    t1 = time.perf_counter()
    return t1 - t0


def benchmark_sglang_text(
    url: str,
    texts: List[str],
    num_requests: int,
    concurrency: int = 1,
) -> BenchmarkResult:
    """Benchmark SGLang text encoding with concurrency."""
    latencies = []
    errors = 0

    # Warmup
    print("  Warming up SGLang text encoding...")
    sglang_encode_text_single(url, texts[0])

    print(f"  Running {num_requests} text encoding requests (concurrency={concurrency})...")
    start_time = time.perf_counter()

    if concurrency == 1:
        # Sequential execution
        for i in range(num_requests):
            text = texts[i % len(texts)]
            try:
                latency = sglang_encode_text_single(url, text)
                latencies.append(latency)
            except Exception as e:
                errors += 1
                print(f"    Error on request {i}: {e}")
    else:
        # Concurrent execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for i in range(num_requests):
                text = texts[i % len(texts)]
                futures.append(executor.submit(sglang_encode_text_single, url, text))

            for future in concurrent.futures.as_completed(futures):
                try:
                    latency = future.result()
                    latencies.append(latency)
                except Exception as e:
                    errors += 1

    total_time = time.perf_counter() - start_time

    return BenchmarkResult(
        name="SGLang",
        mode="text",
        num_requests=num_requests,
        total_time=total_time,
        latencies=latencies,
        errors=errors,
    )


def benchmark_sglang_image(
    url: str,
    image_urls: List[str],
    num_requests: int,
    concurrency: int = 1,
) -> BenchmarkResult:
    """Benchmark SGLang image encoding with concurrency."""
    latencies = []
    errors = 0

    # Warmup
    print("  Warming up SGLang image encoding...")
    sglang_encode_image_single(url, image_urls[0])

    print(f"  Running {num_requests} image encoding requests (concurrency={concurrency})...")
    start_time = time.perf_counter()

    if concurrency == 1:
        # Sequential execution
        for i in range(num_requests):
            img_url = image_urls[i % len(image_urls)]
            try:
                latency = sglang_encode_image_single(url, img_url)
                latencies.append(latency)
            except Exception as e:
                errors += 1
                print(f"    Error on request {i}: {e}")
    else:
        # Concurrent execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for i in range(num_requests):
                img_url = image_urls[i % len(image_urls)]
                futures.append(executor.submit(sglang_encode_image_single, url, img_url))

            for future in concurrent.futures.as_completed(futures):
                try:
                    latency = future.result()
                    latencies.append(latency)
                except Exception as e:
                    errors += 1

    total_time = time.perf_counter() - start_time

    return BenchmarkResult(
        name="SGLang",
        mode="image",
        num_requests=num_requests,
        total_time=total_time,
        latencies=latencies,
        errors=errors,
    )


# =============================================================================
# Reporting
# =============================================================================

def print_result(result: BenchmarkResult):
    """Print a single benchmark result."""
    print(f"\n  {result.name} - {result.mode.upper()} Encoding:")
    print(f"    Requests:        {result.successful_requests}/{result.num_requests} successful")
    print(f"    Total time:      {result.total_time:.2f}s")
    print(f"    Throughput:      {result.throughput:.2f} req/s")
    print(f"    Latency (avg):   {result.avg_latency*1000:.2f}ms")
    print(f"    Latency (p50):   {result.p50_latency*1000:.2f}ms")
    print(f"    Latency (p95):   {result.p95_latency*1000:.2f}ms")
    print(f"    Latency (p99):   {result.p99_latency*1000:.2f}ms")
    print(f"    Latency (min):   {result.min_latency*1000:.2f}ms")
    print(f"    Latency (max):   {result.max_latency*1000:.2f}ms")


def print_comparison(hf_result: Optional[BenchmarkResult], sglang_result: BenchmarkResult):
    """Print comparison between HuggingFace and SGLang results."""
    if hf_result is None:
        return

    print(f"\n  Comparison ({sglang_result.mode.upper()}):")
    print(f"    {'Metric':<20} {'HuggingFace':>15} {'SGLang':>15} {'Speedup':>12}")
    print(f"    {'-'*62}")

    # Throughput comparison
    hf_tp = hf_result.throughput
    sg_tp = sglang_result.throughput
    speedup = sg_tp / hf_tp if hf_tp > 0 else 0
    print(f"    {'Throughput (req/s)':<20} {hf_tp:>15.2f} {sg_tp:>15.2f} {speedup:>11.2f}x")

    # Latency comparison (lower is better, so invert speedup)
    hf_lat = hf_result.avg_latency * 1000
    sg_lat = sglang_result.avg_latency * 1000
    lat_ratio = hf_lat / sg_lat if sg_lat > 0 else 0
    print(f"    {'Avg Latency (ms)':<20} {hf_lat:>15.2f} {sg_lat:>15.2f} {lat_ratio:>11.2f}x")

    hf_p95 = hf_result.p95_latency * 1000
    sg_p95 = sglang_result.p95_latency * 1000
    p95_ratio = hf_p95 / sg_p95 if sg_p95 > 0 else 0
    print(f"    {'P95 Latency (ms)':<20} {hf_p95:>15.2f} {sg_p95:>15.2f} {p95_ratio:>11.2f}x")


def print_summary_table(results: List[BenchmarkResult]):
    """Print a summary table of all results."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    header = f"{'Backend':<12} {'Mode':<8} {'Requests':>10} {'Time (s)':>10} {'Throughput':>12} {'Avg Lat':>10} {'P95 Lat':>10}"
    print(header)
    print("-" * 80)

    for r in results:
        row = f"{r.name:<12} {r.mode:<8} {r.successful_requests:>10} {r.total_time:>10.2f} {r.throughput:>10.2f}/s {r.avg_latency*1000:>9.2f}ms {r.p95_latency*1000:>9.2f}ms"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Benchmark ColQwen3 embedding throughput")
    parser.add_argument("--sglang-url", default=SGLANG_URL, help="SGLang server URL")
    parser.add_argument("--num-requests", type=int, default=50, help="Number of requests per benchmark")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrency level for SGLang")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for HuggingFace")
    parser.add_argument("--skip-hf", action="store_true", help="Skip HuggingFace benchmarks")
    parser.add_argument("--skip-text", action="store_true", help="Skip text encoding benchmarks")
    parser.add_argument("--skip-image", action="store_true", help="Skip image encoding benchmarks")
    args = parser.parse_args()

    print("=" * 80)
    print("ColQwen3 Embedding Throughput Benchmark")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model:         {MODEL_ID}")
    print(f"  Num requests:  {args.num_requests}")
    print(f"  Concurrency:   {args.concurrency} (SGLang)")
    print(f"  Batch size:    {args.batch_size} (HuggingFace)")
    print(f"  Device:        {DEVICE}")

    # Check SGLang server
    print(f"\nChecking SGLang server at {args.sglang_url}...")
    if not check_sglang_server(args.sglang_url):
        print("ERROR: SGLang server is not running!")
        print(f"\nPlease start the server with:")
        print(f"  python -m sglang.launch_server --model {MODEL_ID} \\")
        print("      --trust-remote-code --port 30000 --is-embedding")
        return
    print("  SGLang server is running")

    results = []

    # Preload images for HuggingFace benchmark (avoid network latency)
    images = None
    if not args.skip_hf and not args.skip_image:
        images = preload_images(TEST_IMAGE_URLS)

    # Load HuggingFace model if needed
    hf_model, hf_processor = None, None
    if not args.skip_hf:
        hf_model, hf_processor = get_hf_model_and_processor()

    # =========================================================================
    # Text Encoding Benchmarks
    # =========================================================================
    if not args.skip_text:
        print("\n" + "-" * 80)
        print("TEXT ENCODING BENCHMARKS")
        print("-" * 80)

        # HuggingFace text benchmark
        hf_text_result = None
        if not args.skip_hf:
            print("\n[HuggingFace] Text Encoding Benchmark")
            hf_text_result = benchmark_hf_text(
                hf_model, hf_processor,
                TEST_TEXT_QUERIES,
                args.num_requests,
                args.batch_size,
            )
            print_result(hf_text_result)
            results.append(hf_text_result)

        # SGLang text benchmark
        print("\n[SGLang] Text Encoding Benchmark")
        sglang_text_result = benchmark_sglang_text(
            args.sglang_url,
            TEST_TEXT_QUERIES,
            args.num_requests,
            args.concurrency,
        )
        print_result(sglang_text_result)
        results.append(sglang_text_result)

        # Comparison
        if hf_text_result:
            print_comparison(hf_text_result, sglang_text_result)

    # =========================================================================
    # Image Encoding Benchmarks
    # =========================================================================
    if not args.skip_image:
        print("\n" + "-" * 80)
        print("IMAGE ENCODING BENCHMARKS")
        print("-" * 80)

        # HuggingFace image benchmark
        hf_image_result = None
        if not args.skip_hf and images:
            print("\n[HuggingFace] Image Encoding Benchmark")
            hf_image_result = benchmark_hf_image(
                hf_model, hf_processor,
                images,
                args.num_requests,
                args.batch_size,
            )
            print_result(hf_image_result)
            results.append(hf_image_result)

        # SGLang image benchmark
        print("\n[SGLang] Image Encoding Benchmark")
        sglang_image_result = benchmark_sglang_image(
            args.sglang_url,
            TEST_IMAGE_URLS,
            args.num_requests,
            args.concurrency,
        )
        print_result(sglang_image_result)
        results.append(sglang_image_result)

        # Comparison
        if hf_image_result:
            print_comparison(hf_image_result, sglang_image_result)

    # =========================================================================
    # Summary
    # =========================================================================
    if results:
        print_summary_table(results)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
