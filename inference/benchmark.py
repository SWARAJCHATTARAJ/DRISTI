import time
import json
import numpy as np
from inference.dristi_engine import DristiEngine
from inference.router import DristiRouter

def run_benchmark(iterations: int = 100):
    print("\n" + "=" * 50)
    print("      DRISTI PERFORMANCE BENCHMARK")
    print("=" * 50)
    
    print("Loading Engine and Router...")
    # Initialize components
    engine = DristiEngine()
    router = DristiRouter()
    
    test_q = "Is Python a programming language?"
    test_opts = ["No", "Yes", "Maybe", "Unknown"]
    
    print("\n[1] WARMUP (10 iterations)...")
    for _ in range(10):
        engine.predict(test_q, test_opts)
        
    print(f"\n[2] BENCHMARKING SYSTEM 1 (DristiEngine) - {iterations} iterations...")
    latencies = []
    
    start_total = time.perf_counter()
    for _ in range(iterations):
        start_req = time.perf_counter()
        engine.predict(test_q, test_opts)
        latencies.append((time.perf_counter() - start_req) * 1000) # Convert to ms
    end_total = time.perf_counter()
    
    total_time_s = end_total - start_total
    qps = iterations / total_time_s
    
    avg_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    print("\n--- RESULTS: SYSTEM 1 (FAST PATH) ---")
    print(f"Total Time : {total_time_s:.4f} seconds")
    print(f"Throughput : {qps:.2f} queries per second (QPS)")
    print(f"Avg Latency: {avg_latency:.2f} ms")
    print(f"P50 Latency: {p50_latency:.2f} ms")
    print(f"P95 Latency: {p95_latency:.2f} ms")
    print(f"P99 Latency: {p99_latency:.2f} ms")
    
    print("\n" + "=" * 50)
    print("      ROUTER OVERHEAD BENCHMARK")
    print("=" * 50)
    
    print(f"\n[3] BENCHMARKING PIPELINE/ROUTER - {iterations} iterations...")
    # We use a confident query so it doesn't trigger the 1.5s sleep in System 2
    router_latencies = []
    
    for _ in range(iterations):
        start_req = time.perf_counter()
        router.route_query(test_q, test_opts)
        router_latencies.append((time.perf_counter() - start_req) * 1000)
        
    router_avg = np.mean(router_latencies)
    overhead = router_avg - avg_latency
    
    print("\n--- RESULTS: ROUTER OVERHEAD ---")
    print(f"Router Avg Latency: {router_avg:.2f} ms")
    print(f"Policy/Routing Overhead: {overhead:.2f} ms")
    print("=" * 50)
    print("BENCHMARK COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    run_benchmark(iterations=100)
