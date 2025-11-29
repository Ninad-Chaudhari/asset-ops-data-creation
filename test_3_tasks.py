#!/usr/bin/env python3
"""
Test run with just 3 tasks to verify everything works.
"""

import json
import sys
from pathlib import Path

# Import from main script
from extract_complex_bench import (
    load_json,
    index_tool_schemas,
    build_complexbench_record_for_task,
    call_llm,
)

def main():
    print("🧪 Test run with first 3 tasks")
    print("=" * 60)
    
    # Load data
    print("\n📂 Loading data files...")
    gt_data = load_json("iot_gt.json")
    tool_schema_list = load_json("agent_tool_schemas.json")
    tool_schemas_by_name = index_tool_schemas(tool_schema_list)
    
    # Get first 3 tasks
    tasks = [t for t in gt_data if isinstance(t, dict) and "text" in t][:3]
    print(f"   Processing first {len(tasks)} tasks")
    
    records = []
    
    for i, task in enumerate(tasks, 1):
        task_id = task.get("id", "unknown")
        task_text = task.get("text", "")
        
        print(f"\n📋 Task {i}/3 (ID:{task_id}): {task_text[:60]}...")
        print("   Calling LLM for evaluation questions...")
        
        try:
            record = build_complexbench_record_for_task(
                task,
                tool_schemas_by_name,
                call_llm_fn=call_llm,
            )
            
            # Check results
            annotations = record["complexbench_annotations"]
            success_count = sum(1 for v in annotations.values() if v is not None)
            print(f"   ✓ Generated {success_count}/{len(annotations)} annotation types successfully")
            
            records.append(record)
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Save
    output_path = Path("test_3_tasks.json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Test complete! Saved to {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

