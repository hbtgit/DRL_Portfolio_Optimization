import argparse
import sys
import time

import numpy as np
import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--requests", type=int, default=50)
    ap.add_argument("--max-p95-latency-ms", type=float, required=True)
    args = ap.parse_args()

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}

    meta = requests.get(f"{args.endpoint}/ready", headers=headers, timeout=10)
    meta.raise_for_status()
    window, assets, feats = meta.json()["market_shape"]
    n = meta.json()["num_assets"]

    latencies = []
    for _ in range(args.requests):
        payload = {
            "market": np.random.randn(window, assets, feats).astype(float).tolist(),
            "portfolio": (np.ones(n) / n).tolist(),
        }
        t0 = time.perf_counter()
        resp = requests.post(f"{args.endpoint}/predict", json=payload, headers=headers, timeout=15)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        resp.raise_for_status()
        weights = resp.json()["weights"]
        assert len(weights) == n, f"expected {n} weights, got {len(weights)}"
        assert abs(sum(weights) - 1.0) < 1e-3, f"weights must sum to 1, got {sum(weights)}"

    p95 = float(np.percentile(latencies, 95))
    print(f"p95 latency: {p95:.1f} ms over {args.requests} requests")
    if p95 > args.max_p95_latency_ms:
        print(f"LATENCY SLA FAILED: {p95:.1f}ms > {args.max_p95_latency_ms}ms", file=sys.stderr)
        sys.exit(1)
    print("Integration + latency check passed.")


if __name__ == "__main__":
    main()
