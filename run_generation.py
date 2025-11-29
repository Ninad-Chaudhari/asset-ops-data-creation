#!/usr/bin/env python3
"""
Run ComplexBench generation with progress bar.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

# Import from main script
from extract_complex_bench import (
    load_json,
    index_tool_schemas,
    build_complexbench_record_for_task,
    call_llm,
)

def main():
    print("🚀 Starting ComplexBench Dataset Generation")
    print("=" * 60)
    
    # Load data
    print("\n📂 Loading data files...")
    gt_data = load_json("iot_gt.json")
    tool_schema_list = load_json("agent_tool_schemas.json")
    tool_schemas_by_name = index_tool_schemas(tool_schema_list)
    
    # Filter tasks with "text" field
    tasks = [t for t in gt_data if isinstance(t, dict) and "text" in t]
    total = len(tasks)
    print(f"   Found {total} tasks to process")
    
    records = []
    failed = []
    
    print("\n🤖 Generating evaluation questions with LLM...")
    print("   (This will take approximately 30-60 minutes)")
    print()
    
    for i, task in enumerate(tasks, 1):
        task_id = task.get("id", "unknown")
        task_text = task.get("text", "")[:50] + "..."
        
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
                
        except Exception as e:
            print(f"\n   ⚠️  Error processing task {task_id}: {e}")
            failed.append(task_id)
            # Create a record anyway with None annotations
            record = {
                "id": task.get("uuid") or task.get("id"),
                "original_task": {
                    "text": task.get("text", ""),
                    "characteristic_form": task.get("characteristic_form", ""),
                    "category": task.get("category", ""),
                    "composition_hint": "unknown",
                },
                "complexbench_metadata": {},
                "complexbench_annotations": {
                    "constraint_and_composition": None,
                    "helpfulness_questions": None,
                    "lexical_questions": None,
                    "factuality_questions": None,
                    "chain_tool_questions": None,
                    "argument_questions": None,
                    "consistency_questions": None,
                },
            }
            records.append(record)
    
    # Final progress bar
    print(f"\r   [{'█' * bar_length}] 100.0% | Task {total}/{total} - Complete!    ")
    
    # Save results
    output_path = Path("complexbench_iot_records.json")
    print(f"\n💾 Saving results to {output_path}...")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Generation Complete!")
    print(f"   Total tasks processed: {len(records)}")
    print(f"   Successfully generated: {len(records) - len(failed)}")
    if failed:
        print(f"   Failed or partial: {len(failed)}")
        print(f"   Failed task IDs: {failed[:10]}{'...' if len(failed) > 10 else ''}")
    print(f"   Output file: {output_path}")
    print(f"   File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

