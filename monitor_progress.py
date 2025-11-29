#!/usr/bin/env python3
"""
Monitor the progress of ComplexBench generation.
"""

import json
import time
from pathlib import Path
import sys

output_file = Path("complexbench_iot_records.json")
log_file = Path("generation_log.txt")

print("🔍 Monitoring ComplexBench Generation Progress")
print("=" * 60)

while True:
    try:
        # Check if process is running
        import subprocess
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        is_running = "extract_complex_bench.py" in result.stdout
        
        print(f"\n⏰ {time.strftime('%H:%M:%S')}")
        print(f"   Process Status: {'🟢 RUNNING' if is_running else '🔴 NOT RUNNING'}")
        
        # Check output file
        if output_file.exists():
            try:
                with open(output_file) as f:
                    data = json.load(f)
                
                # Count how many have non-null annotations
                completed = sum(
                    1 for task in data 
                    if task.get("complexbench_annotations", {}).get("helpfulness_questions") is not None
                )
                total = len(data)
                
                print(f"   Progress: {completed}/{total} tasks ({completed/total*100:.1f}%)")
                
                if completed > 0:
                    # Estimate time remaining
                    # Rough estimate: 6 seconds per task (conservative)
                    remaining = (total - completed) * 6 / 60
                    print(f"   Estimated time remaining: ~{remaining:.1f} minutes")
                
            except json.JSONDecodeError:
                print("   Output file exists but not yet valid JSON...")
        else:
            print("   Output file: Not created yet")
        
        # Check log file
        if log_file.exists():
            size = log_file.stat().st_size
            print(f"   Log file size: {size} bytes")
        
        if not is_running:
            print("\n✅ Process completed!")
            if output_file.exists():
                with open(output_file) as f:
                    data = json.load(f)
                print(f"   Final output: {len(data)} tasks processed")
            break
        
        time.sleep(15)  # Check every 15 seconds
        
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
        time.sleep(15)

