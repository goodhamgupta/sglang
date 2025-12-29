"""
SGLang ColQwen3 Multimodal Embeddings Throughput Benchmark

This script benchmarks SGLang's /v1/embeddings API throughput for the ColQwen3
multimodal embedding model, measuring tokens per second.

Key Metrics:
- Embedding tokens per second (primary metric for comparison)
- Request latency (p50, p90, p99)
- Achieved RPS

Usage:
1. First, start the SGLang server with:
   python -m sglang.launch_server --model TomoroAI/tomoro-colqwen3-embed-4b \
       --trust-remote-code --port 30000 --is-embedding

2. Then run this script:
   # Image embedding throughput benchmark (main use case)
   python bench_colqwen3_embeddings.py --mode image

   # Text-only benchmark
   python bench_colqwen3_embeddings.py --mode text

   # Custom settings
   python bench_colqwen3_embeddings.py --mode image --rps 200 --duration 60
"""

import argparse
import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Callable, List

import aiohttp
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

###############################################################################
# CONFIG
###############################################################################
# HTTP Configuration
HTTP_URL = "http://localhost:30000/v1/embeddings"

# ColQwen3 Model Config
COLQWEN3_MODEL_PATH = "TomoroAI/tomoro-colqwen3-embed-4b"

# Text query configuration
TEXT_QUERIES = [
    "What is shown in this image?",
    "Describe the main objects in this picture.",
    "What colors are visible in this image?",
    "Is there any text visible in this image?",
    "What is the setting or location shown?",
    "Describe the composition of this image.",
    "What actions or activities are depicted?",
    "What is the overall mood of this image?",
    "Identify any people or animals in this image.",
    "What time of day does this appear to be?",
]

# Sample image URLs for benchmarking (using picsum for reliable test images)
TEST_IMAGE_URLS = [
    "https://picsum.photos/id/1/800/600",
    "https://picsum.photos/id/10/800/600",
    "https://picsum.photos/id/100/800/600",
    "https://picsum.photos/id/1000/800/600",
    "https://picsum.photos/id/1001/800/600",
    "https://picsum.photos/id/1002/800/600",
    "https://picsum.photos/id/1003/800/600",
    "https://picsum.photos/id/1004/800/600",
    "https://picsum.photos/id/1005/800/600",
    "https://picsum.photos/id/1006/800/600",
]


@dataclass
class BenchmarkResult:
    """Result from a single request."""

    request_id: int
    success: bool
    latency_ms: float
    num_embedding_tokens: int  # Number of tokens in the embedding output
    start_time: float
    end_time: float


@dataclass
class ThroughputStats:
    """Aggregated throughput statistics."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_embedding_tokens: int = 0
    total_duration_secs: float = 0.0
    tokens_per_second: float = 0.0
    requests_per_second: float = 0.0
    latencies_ms: List[float] = field(default_factory=list)

    def compute_stats(self):
        """Compute derived statistics."""
        if self.total_duration_secs > 0:
            self.tokens_per_second = self.total_embedding_tokens / self.total_duration_secs
            self.requests_per_second = self.successful_requests / self.total_duration_secs

    @property
    def p50_latency_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 50)) if self.latencies_ms else 0.0

    @property
    def p90_latency_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 90)) if self.latencies_ms else 0.0

    @property
    def p99_latency_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 99)) if self.latencies_ms else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return mean(self.latencies_ms) if self.latencies_ms else 0.0


###############################################################################
# REQUEST BUILDERS
###############################################################################
def get_random_text_query() -> str:
    """Get a random text query from the pool."""
    return random.choice(TEXT_QUERIES)


def get_random_image_url() -> str:
    """Get a random image URL from the pool."""
    return random.choice(TEST_IMAGE_URLS)


def build_text_request(batch_size: int = 1) -> dict:
    """Build a text-only embeddings request for ColQwen3."""
    if batch_size == 1:
        input_data = get_random_text_query()
    else:
        input_data = [get_random_text_query() for _ in range(batch_size)]

    return {
        "input": input_data,
        "model": COLQWEN3_MODEL_PATH,
    }


def build_image_request(batch_size: int = 1) -> dict:
    """Build an image+text multimodal embeddings request for ColQwen3."""
    # Multimodal input must always be a list of dicts
    input_data = [
        {
            "text": get_random_text_query(),
            "image": get_random_image_url(),
        }
        for _ in range(batch_size)
    ]

    return {
        "input": input_data,
        "model": COLQWEN3_MODEL_PATH,
    }


def count_embedding_tokens(response_data: dict) -> int:
    """
    Count the total number of embedding tokens in the response.

    ColQwen3 returns multi-vector embeddings with shape [num_tokens, embedding_dim].
    Response format:
    {
        "data": [
            {"embedding": [[...], [...], ...], "index": 0},  # num_tokens vectors
            ...
        ]
    }
    """
    total_tokens = 0
    data = response_data.get("data", [])

    for item in data:
        embedding = item.get("embedding", [])
        if isinstance(embedding, list) and len(embedding) > 0:
            if isinstance(embedding[0], list):
                # Multi-vector format: [[tok1], [tok2], ...] -> count outer list
                total_tokens += len(embedding)
            else:
                # Single vector format (fallback): [val1, val2, ...] -> count as 1 token
                total_tokens += 1

    return total_tokens


###############################################################################
# HTTP REQUEST LOGIC
###############################################################################
async def make_request(
    session: aiohttp.ClientSession,
    request_id: int,
    request_data: dict,
    http_url: str,
) -> BenchmarkResult:
    """Make a single HTTP request and return the result with token count."""
    start_time = time.perf_counter()

    try:
        async with session.post(
            http_url,
            json=request_data,
            headers={"Content-Type": "application/json"},
        ) as resp:
            resp_text = await resp.text()
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            if resp.status != 200:
                logger.warning(f"Request {request_id} failed: {resp.status} - {resp_text[:200]}")
                return BenchmarkResult(
                    request_id=request_id,
                    success=False,
                    latency_ms=latency_ms,
                    num_embedding_tokens=0,
                    start_time=start_time,
                    end_time=end_time,
                )

            response_data = json.loads(resp_text)
            num_tokens = count_embedding_tokens(response_data)

            return BenchmarkResult(
                request_id=request_id,
                success=True,
                latency_ms=latency_ms,
                num_embedding_tokens=num_tokens,
                start_time=start_time,
                end_time=end_time,
            )

    except Exception as e:
        end_time = time.perf_counter()
        logger.error(f"Request {request_id} error: {e}")
        return BenchmarkResult(
            request_id=request_id,
            success=False,
            latency_ms=(end_time - start_time) * 1000,
            num_embedding_tokens=0,
            start_time=start_time,
            end_time=end_time,
        )


async def run_throughput_benchmark(
    http_url: str,
    build_request_func: Callable[[int], dict],
    target_rps: int,
    duration_secs: int,
    batch_size: int = 1,
    warmup_requests: int = 5,
    distribution: str = "POISSON",
) -> ThroughputStats:
    """
    Run a throughput benchmark measuring tokens per second.

    Args:
        http_url: SGLang server embeddings endpoint
        build_request_func: Function to build request (takes batch_size)
        target_rps: Target requests per second
        duration_secs: Duration of the benchmark in seconds
        batch_size: Number of items per request
        warmup_requests: Number of warmup requests before measurement
        distribution: "POISSON" or "CONSTANT" request distribution

    Returns:
        ThroughputStats with tokens/second and other metrics
    """
    num_requests = int(target_rps * duration_secs)

    print(f"\nBenchmark Configuration:")
    print(f"  Target RPS: {target_rps}")
    print(f"  Duration: {duration_secs}s")
    print(f"  Total requests: {num_requests}")
    print(f"  Batch size: {batch_size}")
    print(f"  Distribution: {distribution}")

    # Pre-generate all requests
    print(f"\nPre-generating {num_requests} requests...")
    requests = [build_request_func(batch_size) for _ in range(num_requests)]

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300)
    ) as session:
        # Warmup phase
        print(f"\nRunning {warmup_requests} warmup requests...")
        for i in range(warmup_requests):
            warmup_req = build_request_func(batch_size)
            result = await make_request(session, -1, warmup_req, http_url)
            if result.success:
                print(f"  Warmup {i+1}/{warmup_requests}: {result.num_embedding_tokens} tokens, {result.latency_ms:.1f}ms")
            else:
                print(f"  Warmup {i+1}/{warmup_requests}: FAILED")

        # Main benchmark
        print(f"\nStarting benchmark ({num_requests} requests at {target_rps} RPS)...")
        results: List[BenchmarkResult] = []
        tasks: List[asyncio.Task] = []

        benchmark_start = time.perf_counter()

        with tqdm(total=num_requests, desc="Sending requests", unit="req") as pbar:
            for i, request_data in enumerate(requests):
                task = asyncio.create_task(
                    make_request(session, i, request_data, http_url)
                )
                tasks.append(task)
                pbar.update(1)

                # Throttle based on distribution
                if i < num_requests - 1:
                    if distribution == "CONSTANT":
                        await asyncio.sleep(1.0 / target_rps)
                    else:  # POISSON
                        await asyncio.sleep(random.expovariate(target_rps))

        # Wait for all requests to complete
        print("\nWaiting for all requests to complete...")
        with tqdm(total=len(tasks), desc="Completing requests", unit="req") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)

        benchmark_end = time.perf_counter()

    # Compute statistics
    stats = ThroughputStats()
    stats.total_requests = len(results)
    stats.total_duration_secs = benchmark_end - benchmark_start

    for result in results:
        if result.success:
            stats.successful_requests += 1
            stats.total_embedding_tokens += result.num_embedding_tokens
            stats.latencies_ms.append(result.latency_ms)
        else:
            stats.failed_requests += 1

    stats.compute_stats()

    return stats


def print_results(stats: ThroughputStats, mode: str):
    """Print benchmark results."""
    print("\n" + "=" * 70)
    print(f"BENCHMARK RESULTS ({mode.upper()})")
    print("=" * 70)

    print(f"\n>>> THROUGHPUT: {stats.tokens_per_second:,.0f} tokens/second <<<\n")

    print(f"Requests:")
    print(f"  Total:      {stats.total_requests}")
    print(f"  Successful: {stats.successful_requests}")
    print(f"  Failed:     {stats.failed_requests}")
    print(f"  RPS:        {stats.requests_per_second:.1f}")

    print(f"\nTokens:")
    print(f"  Total embedding tokens: {stats.total_embedding_tokens:,}")
    print(f"  Tokens/second:          {stats.tokens_per_second:,.0f}")

    print(f"\nLatency:")
    print(f"  Average: {stats.avg_latency_ms:.1f} ms")
    print(f"  P50:     {stats.p50_latency_ms:.1f} ms")
    print(f"  P90:     {stats.p90_latency_ms:.1f} ms")
    print(f"  P99:     {stats.p99_latency_ms:.1f} ms")

    print(f"\nDuration: {stats.total_duration_secs:.1f} seconds")
    print("=" * 70)


###############################################################################
# MAIN
###############################################################################
async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ColQwen3 multimodal embeddings throughput (tokens/second)"
    )
    parser.add_argument(
        "--mode",
        choices=["text", "image"],
        default="image",
        help="Benchmark mode: text (text-only) or image (image+text multimodal)",
    )
    parser.add_argument(
        "--rps",
        type=int,
        default=100,
        help="Target requests per second (default: 100)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Benchmark duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of items per request (default: 1)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup requests (default: 5)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=HTTP_URL,
        help=f"SGLang server URL (default: {HTTP_URL})",
    )
    parser.add_argument(
        "--distribution",
        choices=["POISSON", "CONSTANT"],
        default="POISSON",
        help="Request distribution pattern (default: POISSON)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("ColQwen3 Multimodal Embeddings THROUGHPUT Benchmark")
    print("=" * 70)
    print(f"Model: {COLQWEN3_MODEL_PATH}")
    print(f"Server URL: {args.url}")
    print(f"Mode: {args.mode}")
    print(f"Target RPS: {args.rps}")
    print(f"Duration: {args.duration}s")
    print(f"Batch size: {args.batch_size}")
    print(f"Distribution: {args.distribution}")

    # Select request builder based on mode
    if args.mode == "text":
        build_func = build_text_request
        print("\nRunning TEXT-ONLY embedding benchmark...")
    else:
        build_func = build_image_request
        print("\nRunning IMAGE+TEXT multimodal embedding benchmark...")

    # Run benchmark
    stats = await run_throughput_benchmark(
        http_url=args.url,
        build_request_func=build_func,
        target_rps=args.rps,
        duration_secs=args.duration,
        batch_size=args.batch_size,
        warmup_requests=args.warmup,
        distribution=args.distribution,
    )

    # Print results
    print_results(stats, args.mode)

    # Final summary line for easy comparison
    print(f"\n*** THROUGHPUT: {stats.tokens_per_second:,.0f} tokens/sec "
          f"(compare with 16K baseline) ***\n")


if __name__ == "__main__":
    asyncio.run(main())
