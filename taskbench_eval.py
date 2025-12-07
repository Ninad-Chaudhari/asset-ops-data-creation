#!/usr/bin/env python3
"""
TaskBench-style Evaluation Script for IoT Agent Tasks

Computes metrics from execution traces:
1. Task Decomposition: ROUGE-1, ROUGE-2, ROUGE-L
2. Tool Selection: Node F1, Edge F1, NED, accuracies
3. Parameter Prediction: t-F1, v-F1

No LLM needed - direct computation from traces!
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple


# ============================================================================
# ROUGE METRICS (for task decomposition)
# ============================================================================

def _tokenize(text: str) -> List[str]:
    """Simple tokenization by splitting on whitespace."""
    return text.lower().split()


def _ngrams(tokens: List[str], n: int) -> List[str]:
    """Generate n-grams from token list."""
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def _f1_from_counts(overlap: int, pred_total: int, gold_total: int) -> float:
    """Compute F1 from overlap counts."""
    if pred_total == 0 or gold_total == 0:
        return 0.0
    precision = overlap / pred_total
    recall = overlap / gold_total
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_n(pred_text: str, gold_text: str, n: int = 1) -> float:
    """Compute ROUGE-N score between predicted and gold text."""
    p_tokens = _tokenize(pred_text)
    g_tokens = _tokenize(gold_text)
    
    pred_ngrams = Counter(_ngrams(p_tokens, n))
    gold_ngrams = Counter(_ngrams(g_tokens, n))
    
    overlap = sum((pred_ngrams & gold_ngrams).values())
    return _f1_from_counts(overlap, sum(pred_ngrams.values()), sum(gold_ngrams.values()))


def _lcs_length(xs: List[str], ys: List[str]) -> int:
    """Compute longest common subsequence length using DP."""
    m, n = len(xs), len(ys)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if xs[i] == ys[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[m][n]


def rouge_l(pred_text: str, gold_text: str) -> float:
    """Compute ROUGE-L score based on longest common subsequence."""
    p_tokens = _tokenize(pred_text)
    g_tokens = _tokenize(gold_text)
    
    if not p_tokens or not g_tokens:
        return 0.0
    
    lcs = _lcs_length(p_tokens, g_tokens)
    precision = lcs / len(p_tokens)
    recall = lcs / len(g_tokens)
    
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ============================================================================
# GRAPH & SET METRICS (for tool selection)
# ============================================================================

def f1_over_sets(pred_set: Set, gold_set: Set) -> float:
    """Compute F1 score between two sets."""
    pred_set = set(pred_set)
    gold_set = set(gold_set)
    
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    
    inter = len(pred_set & gold_set)
    precision = inter / len(pred_set)
    recall = inter / len(gold_set)
    
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def levenshtein(a: List[str], b: List[str]) -> int:
    """Compute Levenshtein edit distance between two sequences."""
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    
    for i in range(len(a)+1):
        dp[i][0] = i
    for j in range(len(b)+1):
        dp[0][j] = j
    
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # delete
                dp[i][j-1] + 1,      # insert
                dp[i-1][j-1] + cost  # substitute
            )
    return dp[-1][-1]


def is_chain(links: List[Dict], step_names: List[str]) -> bool:
    """Check if the execution graph is a simple chain (linear)."""
    if len(step_names) <= 1:
        return True
    
    deg = Counter()
    for l in links:
        deg[l["source"]] += 1
        deg[l["target"]] += 1
    
    # All nodes should appear
    for s in step_names:
        deg.setdefault(s, 0)
    
    # For a chain: exactly 2 endpoints with degree 1, rest with degree 2
    endpoints = sum(1 for v in deg.values() if v == 1)
    return endpoints == 2 and all(v <= 2 for v in deg.values())


def chain_sequence(steps: List[Dict], links: List[Dict]) -> List[str]:
    """Derive sequence of tool actions from a chain (linear) graph."""
    name_to_step = {s["name"]: s for s in steps}
    
    if not links:
        # Single node
        return [steps[0]["action"]] if steps else []
    
    # Build adjacency
    succ = defaultdict(list)
    pred = defaultdict(list)
    for l in links:
        succ[l["source"]].append(l["target"])
        pred[l["target"]].append(l["source"])
    
    # Find start (no predecessors)
    candidates = [n for n in name_to_step if not pred[n]]
    if not candidates:
        return []
    
    start = candidates[0]
    seq = []
    cur = start
    visited = set()
    
    while cur and cur not in visited:
        visited.add(cur)
        seq.append(name_to_step[cur]["action"])
        nexts = succ.get(cur, [])
        cur = nexts[0] if nexts else None
    
    return seq


# ============================================================================
# METRIC COMPUTATION FOR ONE TASK
# ============================================================================

def compute_task_metrics(gt_item: Dict, pred_item: Dict) -> Dict:
    """
    Compute all TaskBench metrics for a single task.
    
    Args:
        gt_item: Ground truth task from iot_gt.json
        pred_item: Predicted task with same structure
    
    Returns:
        Dictionary with all computed metrics
    """
    result = {
        "id": gt_item.get("id") or gt_item.get("uuid"),
        "text": gt_item.get("text", "")[:100] + "...",
        "task_decomposition": {},
        "tool_selection": {},
        "parameter_prediction": {},
    }
    
    # ========================================================================
    # TASK DECOMPOSITION: ROUGE-1, ROUGE-2, ROUGE-L
    # ========================================================================
    
    gt_steps = gt_item.get("planning_steps", [])
    pred_steps = pred_item.get("pred_planning_steps", [])
    
    if gt_steps and pred_steps:
        # Extract instruction text from planning steps
        gt_text = " ".join([
            step.get("instruction", "") if isinstance(step, dict) else str(step)
            for step in gt_steps
        ])
        pred_text = " ".join([
            step.get("instruction", "") if isinstance(step, dict) else str(step)
            for step in pred_steps
        ])
        
        result["task_decomposition"] = {
            "rouge1": round(rouge_n(pred_text, gt_text, n=1), 4),
            "rouge2": round(rouge_n(pred_text, gt_text, n=2), 4),
            "rougeL": round(rouge_l(pred_text, gt_text), 4),
        }
    
    # ========================================================================
    # TOOL SELECTION: Node F1, Edge F1, NED, Accuracies
    # ========================================================================
    
    gt_exec = gt_item.get("execution_steps", [])
    gt_links = gt_item.get("execution_links", [])
    pred_exec = pred_item.get("pred_execution_steps", [])
    pred_links = pred_item.get("pred_execution_links", [])
    
    if gt_exec and pred_exec:
        # Node sets (tool actions, excluding Finish)
        gt_nodes = {s["action"] for s in gt_exec if s["action"] != "Finish"}
        pred_nodes = {s["action"] for s in pred_exec if s["action"] != "Finish"}
        
        node_f1 = f1_over_sets(pred_nodes, gt_nodes)
        node_set_acc = 1.0 if gt_nodes == pred_nodes else 0.0
        
        # Build edge sets at tool-label level
        gt_name_to_action = {s["name"]: s["action"] for s in gt_exec}
        pred_name_to_action = {s["name"]: s["action"] for s in pred_exec}
        
        def edge_set(links, name2act):
            edges = set()
            for l in links:
                s = name2act.get(l["source"])
                t = name2act.get(l["target"])
                if s is not None and t is not None:
                    # Skip edges to/from Finish
                    if s != "Finish" and t != "Finish":
                        edges.add((s, t))
            return edges
        
        gt_edges = edge_set(gt_links, gt_name_to_action)
        pred_edges = edge_set(pred_links, pred_name_to_action)
        
        edge_f1 = f1_over_sets(pred_edges, gt_edges)
        edge_set_acc = 1.0 if gt_edges == pred_edges else 0.0
        graph_acc = 1.0 if (gt_nodes == pred_nodes and gt_edges == pred_edges) else 0.0
        
        tool_sel = {
            "node_f1": round(node_f1, 4),
            "edge_f1": round(edge_f1, 4),
            "node_set_accuracy": round(node_set_acc, 4),
            "edge_set_accuracy": round(edge_set_acc, 4),
            "graph_accuracy": round(graph_acc, 4),
        }
        
        # NED (Normalized Edit Distance) only for chain structures
        step_names = [s["name"] for s in gt_exec if s["action"] != "Finish"]
        if is_chain(gt_links, step_names):
            gold_seq = chain_sequence(gt_exec, gt_links)
            pred_seq = chain_sequence(pred_exec, pred_links)
            
            # Filter out Finish from sequences
            gold_seq = [x for x in gold_seq if x != "Finish"]
            pred_seq = [x for x in pred_seq if x != "Finish"]
            
            if gold_seq and pred_seq:
                dist = levenshtein(gold_seq, pred_seq)
                denom = max(len(gold_seq), len(pred_seq))
                tool_sel["normalized_edit_distance"] = round(dist / denom, 4)
        
        result["tool_selection"] = tool_sel
    
    # ========================================================================
    # PARAMETER PREDICTION: t-F1 and v-F1
    # ========================================================================
    
    if gt_exec and pred_exec:
        def normalize_val(v):
            """Normalize parameter values for comparison."""
            if isinstance(v, str):
                return v.strip().lower()
            return json.dumps(v, sort_keys=True)
        
        # Gold parameter sets
        gt_t = set()  # (tool, param_name) pairs
        gt_v = set()  # (tool, param_name, param_value) triples
        
        for s in gt_exec:
            tool = s["action"]
            args = s.get("arguments", {})
            if isinstance(args, dict):
                for pname, pval in args.items():
                    gt_t.add((tool, pname))
                    gt_v.add((tool, pname, normalize_val(pval)))
        
        # Predicted parameter sets
        pred_t = set()
        pred_v = set()
        
        for s in pred_exec:
            tool = s["action"]
            args = s.get("arguments", {})
            if isinstance(args, dict):
                for pname, pval in args.items():
                    pred_t.add((tool, pname))
                    pred_v.add((tool, pname, normalize_val(pval)))
        
        t_f1 = f1_over_sets(pred_t, gt_t)
        v_f1 = f1_over_sets(pred_v, gt_v)
        
        result["parameter_prediction"] = {
            "t_f1": round(t_f1, 4),
            "v_f1": round(v_f1, 4),
        }
    
    return result


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def evaluate_all(
    gt_path: str = "iot_gt.json",
    pred_path: str = "predictions.json",
    out_path: str = "taskbench_eval.json"
) -> Dict:
    """
    Evaluate all tasks and compute aggregate metrics.
    
    Args:
        gt_path: Path to ground truth file
        pred_path: Path to predictions file
        out_path: Path to save evaluation results
    
    Returns:
        Dictionary with per-task and aggregate metrics
    """
    print("🚀 TaskBench Evaluation")
    print("=" * 60)
    
    # Load data
    print(f"\n📂 Loading ground truth from: {gt_path}")
    with open(gt_path, "r") as f:
        gt_data = json.load(f)
    
    print(f"📂 Loading predictions from: {pred_path}")
    with open(pred_path, "r") as f:
        pred_data = json.load(f)
    
    # Index predictions by id
    pred_index = {}
    for item in pred_data:
        key = item.get("id") or item.get("uuid")
        pred_index[key] = item
    
    print(f"\n✓ Ground truth tasks: {len(gt_data)}")
    print(f"✓ Prediction tasks: {len(pred_data)}")
    
    # Compute metrics for each task
    all_results = []
    agg = {
        "task_decomposition": defaultdict(list),
        "tool_selection": defaultdict(list),
        "parameter_prediction": defaultdict(list),
    }
    
    print(f"\n🔄 Computing metrics...")
    for i, gt_item in enumerate(gt_data, 1):
        # Skip non-task records (metadata)
        if "text" not in gt_item or "id" not in gt_item:
            continue
        
        key = gt_item.get("id") or gt_item.get("uuid")
        pred_item = pred_index.get(key, {})
        
        if not pred_item:
            print(f"   ⚠️  No prediction found for task {key}")
            continue
        
        res = compute_task_metrics(gt_item, pred_item)
        all_results.append(res)
        
        # Aggregate
        for k, v in res.get("task_decomposition", {}).items():
            agg["task_decomposition"][k].append(v)
        for k, v in res.get("tool_selection", {}).items():
            agg["tool_selection"][k].append(v)
        for k, v in res.get("parameter_prediction", {}).items():
            agg["parameter_prediction"][k].append(v)
        
        if i % 20 == 0:
            print(f"   Processed {i}/{len(gt_data)} tasks...")
    
    # Compute averages
    agg_avg = {}
    for section, metrics in agg.items():
        agg_avg[section] = {
            k: round(sum(vals) / len(vals), 4) if vals else None
            for k, vals in metrics.items()
        }
    
    # Prepare output
    output = {
        "per_task": all_results,
        "aggregate": agg_avg,
        "summary": {
            "total_tasks_evaluated": len(all_results),
            "total_tasks_in_ground_truth": len([t for t in gt_data if "text" in t and "id" in t]),
        }
    }
    
    # Save results
    print(f"\n💾 Saving results to: {out_path}")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ TaskBench Evaluation Complete!")
    print("=" * 60)
    print(f"\n📊 Aggregate Metrics:")
    print(f"\n  Task Decomposition:")
    for k, v in agg_avg.get("task_decomposition", {}).items():
        print(f"    {k:20s}: {v:.4f}" if v is not None else f"    {k:20s}: N/A")
    print(f"\n  Tool Selection:")
    for k, v in agg_avg.get("tool_selection", {}).items():
        print(f"    {k:25s}: {v:.4f}" if v is not None else f"    {k:25s}: N/A")
    print(f"\n  Parameter Prediction:")
    for k, v in agg_avg.get("parameter_prediction", {}).items():
        print(f"    {k:20s}: {v:.4f}" if v is not None else f"    {k:20s}: N/A")
    print("\n" + "=" * 60)
    
    return output


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        gt_path = sys.argv[1]
        pred_path = sys.argv[2] if len(sys.argv) > 2 else "predictions.json"
        out_path = sys.argv[3] if len(sys.argv) > 3 else "taskbench_eval.json"
    else:
        gt_path = "iot_gt.json"
        pred_path = "predictions.json"
        out_path = "taskbench_eval.json"
    
    evaluate_all(gt_path, pred_path, out_path)

