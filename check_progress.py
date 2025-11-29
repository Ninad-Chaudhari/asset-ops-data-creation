#!/usr/bin/env python3
"""Quick check of generation progress"""
import json
import time
from pathlib import Path

output_file = Path("/Users/ninad/Desktop/NYU/Dynamic_Evaluation/complex-bench-pipeline/complexbench_iot_records.json")

print("Checking generation progress...")
print()

try:
    if output_file.exists():
        with open(output_file) as f:
            data = json.load(f)
        
        total = len(data)
        with_data = sum(1 for t in data if t.get("complexbench_annotations", {}).get("helpfulness_questions") is not None)
        
        print(f"✅ Found output file")
        print(f"   Total records: {total}")
        print(f"   With LLM data: {with_data}")
        print(f"   Progress: {with_data}/{total} ({with_data/total*100:.1f}%)")
        
        if with_data == 0:
            print()
            print("⚠️  No LLM data yet - generation might just be starting...")
        elif with_data < total:
            remaining = (total - with_data) * 5 / 60  # ~5 sec per task
            print(f"   Estimated time remaining: ~{remaining:.1f} minutes")
        else:
            print()
            print("🎉 Generation complete!")
            
    else:
        print("⏳ Output file not created yet - generation starting...")
except Exception as e:
    print(f"❌ Error: {e}")

