#!/usr/bin/env python3
"""
CORRECTED: Convert enriched trajectory files to TaskBench predictions format.

Key fix: Extract tools from react_iterations, not tool_calls
The react_iterations contain the actual sequence of actions the agent took.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple


def parse_plan_text(plan_text: str) -> List[str]:
    """
    Extract planning steps from the plan_text.
    
    Example input:
        1. #Task1: Identify Chiller 6
           #Agent1: IoT Data Download
           ...
    
    Returns list of task descriptions.
    """
    steps = []
    lines = plan_text.strip().split('\n')
    current_task = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if this is a task line (starts with number)
        if re.match(r'^\d+\.', line):
            if current_task:
                steps.append(' '.join(current_task))
            current_task = []
        
        # Extract task description
        if '#Task' in line:
            match = re.search(r'#Task\d*:\s*(.+)', line)
            if match:
                current_task.append(match.group(1))
    
    # Add last task
    if current_task:
        steps.append(' '.join(current_task))
    
    return steps


def parse_action_input(action_input: str) -> Dict[str, Any]:
    """
    Parse action_input string into dictionary.
    
    Examples:
        - "{}" → {}
        - "site_name=MAIN" → {"site_name": "MAIN"}
        - "site_name=MAIN, assetnum=Chiller 6" → {"site_name": "MAIN", "assetnum": "Chiller 6"}
        - '{"site": "MAIN"}' → {"site": "MAIN"}
    """
    if not action_input or action_input.strip() == "{}":
        return {}
    
    # Try parsing as JSON first
    try:
        return json.loads(action_input)
    except:
        pass
    
    # Parse key=value format
    args = {}
    if '=' in action_input:
        # Handle comma-separated key=value pairs
        parts = action_input.split(',')
        for part in parts:
            if '=' in part:
                key, value = part.strip().split('=', 1)
                # Clean up quotes
                value = value.strip().strip('"').strip("'")
                args[key.strip()] = value
    
    return args


def extract_execution_from_trace(trace: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Extract execution steps and links from the enriched trajectory.
    
    KEY FIX: Use react_iterations to get the actual action sequence!
    """
    pred_execution_steps = []
    pred_execution_links = []
    
    step_counter = 0
    previous_step_name = None
    
    for step_idx, step in enumerate(trace):
        exec_details = step.get('execution_details', {})
        react_iterations = exec_details.get('react_iterations', [])
        
        # Process each react iteration (these are the actual actions!)
        for react_iter in react_iterations:
            action = react_iter.get('action', '')
            action_input = react_iter.get('action_input', '')
            
            # Skip Finish actions for now
            if action == 'Finish':
                continue
            
            step_counter += 1
            step_name = f"step{step_counter}"
            
            # Parse arguments
            arguments = parse_action_input(action_input)
            
            pred_execution_steps.append({
                'name': step_name,
                'action': action,
                'arguments': arguments
            })
            
            # Create links (sequential for now)
            if previous_step_name:
                pred_execution_links.append({
                    'source': previous_step_name,
                    'target': step_name
                })
            
            previous_step_name = step_name
    
    # Add final Finish step
    if pred_execution_steps:
        pred_execution_steps.append({
            'name': 'finish',
            'action': 'Finish',
            'arguments': {}
        })
        pred_execution_links.append({
            'source': previous_step_name,
            'target': 'finish'
        })
    
    return pred_execution_steps, pred_execution_links


def convert_trajectory_to_prediction(trajectory: Dict) -> Dict:
    """
    Convert a single trajectory file to TaskBench prediction format.
    
    Args:
        trajectory: Dict loaded from Q_XXX_trajectory.json
    
    Returns:
        Prediction dict for taskbench_eval.py
    """
    task_id = trajectory.get('id')
    enriched = trajectory.get('enriched_trajectory', {})
    
    # Extract planning steps
    plan_gen = enriched.get('plan_generation')
    pred_planning_steps = []
    
    if plan_gen and isinstance(plan_gen, dict):
        plan_text = plan_gen.get('plan_text', '')
        if plan_text:
            pred_planning_steps = parse_plan_text(plan_text)
    
    # If no planning, extract from execution trace task descriptions
    if not pred_planning_steps:
        exec_trace = enriched.get('execution_trace', [])
        pred_planning_steps = [
            step.get('task_description', '')
            for step in exec_trace
            if step.get('task_description')
        ]
    
    # Extract execution steps and links from react iterations
    exec_trace = enriched.get('execution_trace', [])
    pred_execution_steps, pred_execution_links = extract_execution_from_trace(exec_trace)
    
    return {
        'id': task_id,
        'pred_planning_steps': pred_planning_steps,
        'pred_execution_steps': pred_execution_steps,
        'pred_execution_links': pred_execution_links
    }


def convert_trajectories_to_predictions(
    trajectory_dir: str = "/Users/ninad/Downloads",
    pattern: str = "Q_*_trajectory.json",
    output_file: str = "predictions_from_trajectories.json"
):
    """
    Convert all trajectory files in a directory to predictions format.
    
    Args:
        trajectory_dir: Directory containing trajectory files
        pattern: Glob pattern for trajectory files
        output_file: Output predictions file
    """
    print("🔄 Converting Trajectory Files to TaskBench Predictions (CORRECTED)")
    print("=" * 60)
    print("KEY FIX: Using react_iterations to extract actual tool sequence")
    print()
    
    trajectory_path = Path(trajectory_dir)
    trajectory_files = sorted(trajectory_path.glob(pattern))
    
    print(f"📂 Found {len(trajectory_files)} trajectory files")
    print(f"   Pattern: {pattern}")
    print(f"   Directory: {trajectory_dir}")
    print()
    
    predictions = []
    failed = []
    
    for traj_file in trajectory_files:
        try:
            with open(traj_file, 'r') as f:
                trajectory = json.load(f)
            
            pred = convert_trajectory_to_prediction(trajectory)
            predictions.append(pred)
            
            task_id = pred['id']
            num_steps = len(pred['pred_execution_steps'])
            print(f"   ✓ Task {task_id}: {num_steps} execution steps extracted")
            
        except Exception as e:
            print(f"   ✗ Failed {traj_file.name}: {e}")
            failed.append(traj_file.name)
            import traceback
            traceback.print_exc()
    
    # Save predictions
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"\n💾 Results:")
    print(f"   Successfully converted: {len(predictions)}")
    print(f"   Failed: {len(failed)}")
    print(f"   Output file: {output_path}")
    
    if failed:
        print(f"\n⚠️  Failed files: {failed}")
    
    print("\n✅ Conversion complete!")
    print("=" * 60)
    print(f"\nNext step: Run TaskBench evaluation")
    print(f"  python3 taskbench_eval.py <ground_truth.json> {output_file} results.json")
    
    return predictions


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        trajectory_dir = sys.argv[1]
        pattern = sys.argv[2] if len(sys.argv) > 2 else "Q_*_trajectory.json"
        output_file = sys.argv[3] if len(sys.argv) > 3 else "predictions_from_trajectories.json"
    else:
        trajectory_dir = "/Users/ninad/Downloads"
        pattern = "Q_*_trajectory.json"
        output_file = "predictions_from_trajectories.json"
    
    convert_trajectories_to_predictions(trajectory_dir, pattern, output_file)

