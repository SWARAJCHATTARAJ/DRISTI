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
    from inference.dual_engine import DualEngineRouter
    router = DualEngineRouter()
    
    test_q = "Is Python a programming language?"
    test_opts = ["No", "Yes", "Maybe", "Unknown"]
    
    print("\n[1] WARMUP (10 iterations)...")
    for _ in range(10):
        router.fast_engine.predict(test_q, test_opts)
        router.heavy_engine.predict(test_q, test_opts)
        
    print(f"\n[2] BENCHMARKING SYSTEM 1 (V1_FAST) - {iterations} iterations...")
    latencies = []
    start_total = time.perf_counter()
    for _ in range(iterations):
        start_req = time.perf_counter()
        router.fast_engine.predict(test_q, test_opts)
        latencies.append((time.perf_counter() - start_req) * 1000)
    end_total = time.perf_counter()
    
    total_time_s = end_total - start_total
    qps = iterations / total_time_s
    avg_latency = np.mean(latencies)
    
    print("\n--- RESULTS: SYSTEM 1 (FAST PATH) ---")
    print(f"Total Time : {total_time_s:.4f} seconds")
    print(f"Throughput : {qps:.2f} queries per second (QPS)")
    print(f"Avg Latency: {avg_latency:.2f} ms")
    
    print(f"\n[3] BENCHMARKING SYSTEM 2 (V2_HEAVY) - {iterations} iterations...")
    latencies_v2 = []
    start_total_v2 = time.perf_counter()
    for _ in range(iterations):
        start_req = time.perf_counter()
        router.heavy_engine.predict(test_q, test_opts)
        latencies_v2.append((time.perf_counter() - start_req) * 1000)
    end_total_v2 = time.perf_counter()
    
    total_time_s_v2 = end_total_v2 - start_total_v2
    qps_v2 = iterations / total_time_s_v2
    avg_latency_v2 = np.mean(latencies_v2)
    
    print("\n--- RESULTS: SYSTEM 2 (HEAVY PATH) ---")
    print(f"Total Time : {total_time_s_v2:.4f} seconds")
    print(f"Throughput : {qps_v2:.2f} queries per second (QPS)")
    print(f"Avg Latency: {avg_latency_v2:.2f} ms")

    print("\n" + "=" * 50)
    print("BENCHMARK COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    run_benchmark(iterations=100)
