#!/usr/bin/env python3
"""
Test script to verify LLM calls work correctly on a single task.
"""

import json
import os
import sys
from pathlib import Path

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set")
    print("Please set it with: export OPENAI_API_KEY='your-key-here'")
    sys.exit(1)

# Import the main script functions
sys.path.insert(0, str(Path(__file__).parent))
from extract_complex_bench import (
    load_json,
    index_tool_schemas,
    build_complexbench_record_for_task,
    call_llm,
)

def test_single_task():
    """Test LLM call on the first task."""
    print("🧪 Testing LLM calls on first task...")
    
    # Load data
    gt_data = load_json("iot_gt.json")
    tool_schema_list = load_json("agent_tool_schemas.json")
    tool_schemas_by_name = index_tool_schemas(tool_schema_list)
    
    # Get first task
    first_task = gt_data[0]
    print(f"\n📋 Task: {first_task['text']}")
    print(f"   ID: {first_task['id']}")
    print(f"   Category: {first_task['category']}")
    
    # Build record with LLM calls
    print("\n🤖 Generating evaluation questions...")
    record = build_complexbench_record_for_task(
        first_task,
        tool_schemas_by_name,
        call_llm_fn=call_llm,
    )
    
    # Check results
    print("\n✅ Results:")
    annotations = record["complexbench_annotations"]
    
    for key, value in annotations.items():
        if value is not None:
            print(f"   ✓ {key}: SUCCESS")
            if isinstance(value, dict) and "questions" in value:
                print(f"      Generated {len(value['questions'])} questions")
        else:
            print(f"   ✗ {key}: FAILED (returned None)")
    
    # Save test output
    output_path = Path("test_output.json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Test output saved to: {output_path}")
    print("\n✨ Test complete! If all checks passed, you can run the full dataset.")
    
    return record

if __name__ == "__main__":
    try:
        test_single_task()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

