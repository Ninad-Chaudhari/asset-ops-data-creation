#!/usr/bin/env python3
"""
Generate ComplexBench dataset for FMSR tasks with incremental updates.
"""

import json
import sys
import os
from pathlib import Path

# Set API key (should be set as environment variable before running)
# export OPENAI_API_KEY="your-key-here"
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY environment variable must be set")

# Import from main script
from extract_complex_bench import (
    load_json,
    index_tool_schemas,
    build_complexbench_record_for_task,
    call_llm,
)

def main():
    print("🚀 Starting ComplexBench Dataset Generation for FMSR")
    print("=" * 60)
    
    # Load data
    print("\n📂 Loading data files...")
    gt_data = load_json("fsmr-gt.json")
    tool_schema_list = load_json("fmsr_tool_schemas.json")
    tool_schemas_by_name = index_tool_schemas(tool_schema_list)
    
    # Filter tasks (skip the metadata record)
    tasks = [t for t in gt_data if isinstance(t, dict) and "text" in t and "id" in t]
    total = len(tasks)
    print(f"   Found {total} FMSR tasks to process (IDs: {tasks[0]['id']}-{tasks[-1]['id']})")
    
    output_path = Path("complexbench_fmsr_records.json")
    records = []
    failed = []
    
    print("\n🤖 Generating evaluation questions with LLM...")
    print("   (This will take approximately 2-3 minutes for 20 tasks)")
    print("   File will update after EACH task!")
    print()
    
    for i, task in enumerate(tasks, 1):
        task_id = task.get("id", "unknown")
        task_text = task.get("text", "")[:60] + "..."
        
        # Progress bar
        percent = (i-1) / total * 100
        bar_length = 40
        filled = int(bar_length * (i-1) / total)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        print(f"\r   [{bar}] {percent:5.1f}% | Task {i}/{total} (ID:{task_id})", end="", flush=True)
        
        try:
            record = build_complexbench_record_for_task(
                task,
                tool_schemas_by_name,
                call_llm_fn=call_llm,
            )
            records.append(record)
            
            # Check if any annotations failed
            annotations = record["complexbench_annotations"]
            if any(v is None for v in annotations.values()):
                failed.append(task_id)
            
            # ✨ INCREMENTAL SAVE
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"\n   ⚠️  Error processing task {task_id}: {e}")
            failed.append(task_id)
            continue
    
    # Final progress bar
    print(f"\r   [{'█' * bar_length}] 100.0% | Task {total}/{total} - Complete!    ")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ FMSR Generation Complete!")
    print(f"   Total tasks processed: {len(records)}")
    print(f"   Successfully generated: {len(records) - len(failed)}")
    if failed:
        print(f"   Failed or partial: {len(failed)}")
        print(f"   Failed task IDs: {failed}")
    print(f"   Output file: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.2f} KB")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
        print("   Partial results have been saved to complexbench_fmsr_records.json")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


