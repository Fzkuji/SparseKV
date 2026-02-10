#!/usr/bin/env python3
"""Wrapper around kvpress evaluate.py that records latency, throughput, and peak GPU memory."""

import subprocess
import sys
import time
import json
import os
import threading

def monitor_gpu_memory(interval=1.0, result={}):
    """Monitor peak GPU memory in background thread."""
    import subprocess
    result["peak_mb"] = 0
    result["running"] = True
    while result["running"]:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
                text=True
            )
            mem_values = [int(x.strip()) for x in out.strip().split("\n") if x.strip()]
            current_max = max(mem_values) if mem_values else 0
            result["peak_mb"] = max(result["peak_mb"], current_max)
        except Exception:
            pass
        time.sleep(interval)

def main():
    # Pass all args to evaluate.py
    eval_args = sys.argv[1:]
    
    # Parse output_dir and build result path for saving profiling info
    output_dir = None
    for i, arg in enumerate(eval_args):
        if arg == "--output_dir" and i + 1 < len(eval_args):
            output_dir = eval_args[i + 1]
    
    # Start GPU memory monitor
    mem_result = {}
    monitor_thread = threading.Thread(target=monitor_gpu_memory, args=(1.0, mem_result), daemon=True)
    monitor_thread.start()
    
    # Run evaluation
    start_time = time.time()
    
    cmd = [sys.executable, "evaluate.py"] + eval_args
    print(f"[eval_wrapper] Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False)
    
    elapsed = time.time() - start_time
    mem_result["running"] = False
    monitor_thread.join(timeout=3)
    
    # Find the most recently modified metrics.json in output_dir
    profile_data = {
        "total_time_seconds": round(elapsed, 2),
        "total_time_minutes": round(elapsed / 60, 2),
        "peak_gpu_memory_mb": mem_result.get("peak_mb", -1),
        "peak_gpu_memory_gb": round(mem_result.get("peak_mb", 0) / 1024, 2),
    }
    
    # Try to compute throughput from predictions
    if output_dir:
        # Find latest result dir
        try:
            subdirs = []
            for d in os.listdir(output_dir):
                pred_path = os.path.join(output_dir, d, "predictions.csv")
                if os.path.exists(pred_path):
                    subdirs.append((os.path.getmtime(pred_path), d))
            if subdirs:
                subdirs.sort(reverse=True)
                latest_dir = subdirs[0][1]
                pred_path = os.path.join(output_dir, latest_dir, "predictions.csv")
                
                # Count samples
                with open(pred_path) as f:
                    n_samples = sum(1 for _ in f) - 1  # minus header
                
                profile_data["num_samples"] = n_samples
                profile_data["seconds_per_sample"] = round(elapsed / max(n_samples, 1), 2)
                profile_data["samples_per_minute"] = round(n_samples / (elapsed / 60), 2)
                
                # Save profiling data alongside metrics
                profile_path = os.path.join(output_dir, latest_dir, "profiling.json")
                with open(profile_path, "w") as f:
                    json.dump(profile_data, f, indent=2)
                print(f"[eval_wrapper] Profiling saved to {profile_path}")
        except Exception as e:
            print(f"[eval_wrapper] Warning: could not save profiling: {e}")
    
    # Always print summary
    print(f"\n[eval_wrapper] === Profiling Summary ===")
    print(f"  Total time:       {profile_data['total_time_minutes']:.1f} min")
    print(f"  Peak GPU memory:  {profile_data.get('peak_gpu_memory_gb', '?')} GB")
    if 'samples_per_minute' in profile_data:
        print(f"  Throughput:       {profile_data['samples_per_minute']:.1f} samples/min")
        print(f"  Latency:          {profile_data['seconds_per_sample']:.2f} s/sample")
    
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
