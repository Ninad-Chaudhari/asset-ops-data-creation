# Multi-Tool Agent Evaluation Framework

A comprehensive evaluation framework for assessing multi-tool agent performance through two complementary approaches: **Answer Quality Evaluation** and **Structural Correctness Evaluation**.

---

## 🚀 Quick Start: Complete Evaluation Pipeline

This section provides all commands needed to run a complete evaluation from trajectory generation to final metrics.

### Prerequisites

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Install dependencies
pip install openai nltk numpy
```

### Step 1: Generate Agent Trajectory

**Using Docker (Recommended)**:

```bash
# Navigate to CODS Track 2
cd ../benchmark/cods_track2

# For FMSR tasks (with ground truth questions)
docker compose up

# Or for specific task IDs
docker compose run assetopsbench python /home/run_track_2.py \
  --utterance_ids 106,107,108 \
  --use_planning \
  --use_fsmr_gt

# Output: track2_result/trajectory/Q_106_trajectory.json
```

**Without Docker** (if agent-hive is installed locally):

```bash
python run_track_2.py --utterance_ids 106 --use_planning --use_fsmr_gt
```

### Step 2: Generate Answer Quality Evaluation Questions

**For IoT Tasks**:

```bash
cd ../../complex-bench-pipeline

# Generate evaluation questions for all IoT tasks
python3 run_generation_incremental.py

# Output: complexbench_iot_records.json
# Contains 6 evaluation dimensions per task:
# - Helpfulness, Lexical, Chain/Tool, Arguments, Factuality, Consistency
```

**For FMSR Tasks**:

```bash
# Generate evaluation questions for all FMSR tasks
python3 run_fmsr_generation.py

# Output: complexbench_fmsr_records.json
```

### Step 3: Convert Trajectory to Evaluation Format

```bash
# Convert single trajectory
python3 convert_trajectories_corrected.py \
  ../benchmark/cods_track2/track2_result/trajectory \
  "Q_106_trajectory.json" \
  predictions_Q106.json

# Convert multiple trajectories (all FMSR tasks)
python3 convert_trajectories_corrected.py \
  ../benchmark/cods_track2/track2_result/trajectory \
  "Q_10*_trajectory.json" \
  predictions_all_fsmr.json

# Output: predictions_Q106.json
# Format:
# [
#   {
#     "id": 106,
#     "pred_planning_steps": [...],
#     "pred_execution_steps": [...],
#     "pred_execution_links": [...]
#   }
# ]
```

### Step 4: Run Structural Correctness Evaluation

```bash
# Evaluate against FMSR ground truth
python3 taskbench_eval.py \
  fsmr-gt.json \
  predictions_Q106.json \
  results_structural_Q106.json

# Evaluate against IoT ground truth
python3 taskbench_eval.py \
  iot_gt.json \
  predictions_iot.json \
  results_structural_iot.json

# Output: results_structural_Q106.json
# Contains:
# - Task Decomposition: ROUGE-1, ROUGE-2, ROUGE-L
# - Tool Selection: Node F1, Edge F1, NED, Accuracy
# - Parameter Prediction: t-F1, v-F1
```

### Step 5: View Results

```bash
# View comprehensive results
cat results_structural_Q106.json | python3 -m json.tool

# View aggregate metrics only
cat results_structural_Q106.json | jq '.aggregate'

# View specific task
cat results_structural_Q106.json | jq '.per_task[] | select(.id==106)'
```

### Complete Pipeline Example (Single Task)

```bash
# 1. Generate trajectory for Task 106
cd ../benchmark/cods_track2
docker compose up

# 2. Generate answer quality questions (if not done)
cd ../../complex-bench-pipeline
python3 run_fmsr_generation.py

# 3. Convert trajectory
python3 convert_trajectories_corrected.py \
  ../benchmark/cods_track2/track2_result/trajectory \
  "Q_106_trajectory.json" \
  predictions_Q106.json

# 4. Evaluate structural correctness
python3 taskbench_eval.py \
  fsmr-gt.json \
  predictions_Q106.json \
  results_Q106.json

# 5. View results
cat results_Q106.json | jq '.aggregate'
```

### Batch Processing (Multiple Tasks)

```bash
# 1. Generate trajectories for tasks 101-120
cd ../benchmark/cods_track2
# Edit entrypoint.sh to set: --utterance_ids 101,102,103,...,120
docker compose up

# 2. Convert all trajectories
cd ../../complex-bench-pipeline
python3 convert_trajectories_corrected.py \
  ../benchmark/cods_track2/track2_result/trajectory \
  "Q_*_trajectory.json" \
  predictions_all_fsmr.json

# 3. Evaluate all
python3 taskbench_eval.py \
  fsmr-gt.json \
  predictions_all_fsmr.json \
  results_all_fsmr.json

# 4. View aggregate metrics across all tasks
cat results_all_fsmr.json | jq '.aggregate'
cat results_all_fsmr.json | jq '.summary'
```

### Expected Outputs

**Answer Quality Metrics** (from complexbench_*_records.json):
- Each task gets 6 evaluation questions
- Use these to assess agent's final answers
- Scores: 1-5 scale or Correct/Incorrect

**Structural Correctness Metrics** (from results_*.json):
```json
{
  "aggregate": {
    "task_decomposition": {
      "rouge1": 0.37,    // Higher is better (0-1)
      "rouge2": 0.19,
      "rougeL": 0.30
    },
    "tool_selection": {
      "node_f1": 0.45,        // F1 of tools used (0-1)
      "edge_f1": 0.38,        // F1 of tool sequences (0-1)
      "graph_accuracy": 0.25, // Exact graph match (0-1)
      "normalized_edit_distance": 0.62 // Lower is better (0-1)
    },
    "parameter_prediction": {
      "t_f1": 0.33,  // Tool+param name correctness (0-1)
      "v_f1": 0.28   // Tool+param+value correctness (0-1)
    }
  }
}
```

### Troubleshooting

**No tool calls extracted**:
```bash
# Check if trajectory has react_iterations
cat ../benchmark/cods_track2/track2_result/trajectory/Q_106_trajectory.json | \
  jq '.enriched_trajectory.execution_trace[].execution_details.react_iterations'
```

**Question mismatch between trajectory and ground truth**:
```bash
# Verify questions match
python3 -c "
import json
traj = json.load(open('../benchmark/cods_track2/track2_result/trajectory/Q_106_trajectory.json'))
gt = json.load(open('fsmr-gt.json'))
gt_task = [t for t in gt if t['id']==106][0]
print('Trajectory:', traj['text'])
print('Ground truth:', gt_task['text'])
print('Match:', traj['text'] == gt_task['text'])
"
```

**Low scores (0%)**:
- Agent may have used different but valid approach
- Check if tools/parameters are semantically equivalent
- Low scores don't always mean poor performance!

---

## Table of Contents

- [Quick Start](#-quick-start-complete-evaluation-pipeline)
- [Overview](#overview)
- [Evaluation Methodology](#evaluation-methodology)
  - [Phase 1: Answer Quality Evaluation](#phase-1-answer-quality-evaluation)
  - [Phase 2: Structural Correctness Evaluation](#phase-2-structural-correctness-evaluation)
- [Pipeline Architecture](#pipeline-architecture)
- [Ground Truth Schema](#ground-truth-schema)
- [Trajectory Generation](#trajectory-generation)
- [Evaluation Execution](#evaluation-execution)
- [Getting Started](#getting-started)
- [Results Interpretation](#results-interpretation)

---

## Overview

This framework evaluates multi-tool agents across two critical dimensions:

1. **Answer Quality**: Does the agent produce helpful, accurate, and consistent responses?
2. **Structural Correctness**: Does the agent decompose tasks correctly, select appropriate tools, and predict parameters accurately?

### Key Features

- **Dual Evaluation**: Combines semantic answer quality with structural process correctness
- **LLM-Powered Quality Assessment**: Uses GPT-4 for nuanced evaluation of answer quality
- **Automated Structural Metrics**: Computes ROUGE, F1, and graph similarity scores
- **Few-Shot Prompting**: Incorporates examples for consistent LLM evaluation
- **Incremental Processing**: Real-time progress tracking for long-running evaluations
- **Domain-Agnostic**: Works with any multi-tool agent system

---

## Evaluation Methodology

### Phase 1: Answer Quality Evaluation

This phase assesses the **semantic quality** of agent responses across six dimensions.

#### Extracted Features

From each ground truth task, we extract:

**1. Task Question (`text`)**
- The original user query
- Example: *"List all failure modes of Chiller 6 at the MAIN site that can be detected by Chiller 6 Supply Temperature."*

**2. Expected Output (`expected_output`)**
- The gold-standard answer
- Used as reference for quality assessment

**3. Tool Context (`tool_schemas`)**
- Schemas of tools involved in the task
- Provides context for understanding tool usage

**4. Execution Trace (`execution_steps`)**
- Ground truth sequence of tool calls
- Used to generate reference execution context

#### Evaluation Dimensions

**1. Helpfulness**
- Does the answer directly address the user's question?
- Is it actionable and useful?
- Scoring: 1-5 scale with detailed justification

**2. Lexical Correctness**
- Are entity names, technical terms, and identifiers correct?
- Are numbers, dates, and units accurate?
- Scoring: Binary (Correct/Incorrect) with explanation

**3. Chain & Tool Usage**
- Is the sequence of tool calls logical and efficient?
- Are the right tools selected for the task?
- Scoring: 1-5 scale based on correctness and efficiency

**4. Argument Correctness**
- Are tool arguments valid and appropriate?
- Do parameters match expected types and values?
- Scoring: 1-5 scale with per-argument analysis

**5. Factuality & Supportiveness**
- Are claims factually correct based on tool outputs?
- Is there evidence to support the answer?
- Scoring: 1-5 scale with factual grounding assessment

**6. Consistency**
- Is the answer internally consistent?
- Do different parts of the response align?
- Scoring: 1-5 scale evaluating coherence

#### Few-Shot Prompting

Each evaluation dimension includes 2-3 example assessments to guide the LLM:

```
Example:
Question: [Sample question]
Expected: [Gold answer]
Predicted: [Agent answer]
Score: [1-5]
Justification: [Detailed reasoning]
```

This ensures consistent and reliable evaluation across all tasks.

#### Output Format

For each task, generates:

```json
{
  "id": 106,
  "text": "List all failure modes...",
  "expected_output": "The following failure modes...",
  "tool_schemas": ["Get Failure Modes", ...],
  "complexbench_annotations": {
    "helpfulness": {"score": 4, "justification": "..."},
    "lexical": {"correct": true, "explanation": "..."},
    "chain_tool": {"score": 5, "justification": "..."},
    "arguments": {"score": 4, "justification": "..."},
    "factuality": {"score": 5, "justification": "..."},
    "consistency": {"score": 5, "justification": "..."}
  }
}
```

---

### Phase 2: Structural Correctness Evaluation

This phase assesses the **structural correctness** of agent execution through three categories of metrics.

#### Extracted Features

From each ground truth task, we extract:

**1. Planning Steps (`planning_steps`)**
- Natural language description of sub-tasks
- Example: 
  ```json
  ["Identify the site and asset", 
   "Retrieve failure modes for Chiller 6",
   "Map failure modes to sensor"]
  ```

**2. Execution Steps (`execution_steps`)**
- Structured tool call specifications
- Example:
  ```json
  [
    {"name": "get_failure_modes", "action": "Get Failure Modes", 
     "arguments": "Chiller 6"},
    {"name": "get_mapping", "action": "Get Failure Mode and Sensor Relevancy Mapping",
     "arguments": "Chiller 6, <chiller_failure_list>, Chiller 6 Supply Temperature"}
  ]
  ```

**3. Execution Links (`execution_links`)**
- Dependencies between execution steps
- Example:
  ```json
  [
    {"source": "get_failure_modes", "target": "get_mapping"}
  ]
  ```

#### Evaluation Metrics

**Category 1: Task Decomposition**

Measures how well the agent breaks down complex tasks into sub-tasks.

- **ROUGE-1**: Unigram overlap between predicted and ground truth planning steps
- **ROUGE-2**: Bigram overlap for capturing phrase-level similarity
- **ROUGE-L**: Longest common subsequence for assessing structural similarity

**Computation**:
```python
def rouge_n(pred_tokens, gold_tokens, n):
    pred_ngrams = set(ngrams(pred_tokens, n))
    gold_ngrams = set(ngrams(gold_tokens, n))
    
    overlap = len(pred_ngrams & gold_ngrams)
    recall = overlap / len(gold_ngrams) if gold_ngrams else 0
    precision = overlap / len(pred_ngrams) if pred_ngrams else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}
```

**Category 2: Tool Selection**

Evaluates whether the agent selects the correct tools and sequences them properly.

- **Node F1**: F1 score of tool names (treats execution as a set of tools)
- **Edge F1**: F1 score of tool transitions (captures execution order)
- **Node Set Accuracy**: Exact match of tool sets
- **Edge Set Accuracy**: Exact match of tool transition sets
- **Graph Accuracy**: Exact match of complete execution graph
- **Normalized Edit Distance (NED)**: Levenshtein distance between tool sequences

**Computation**:
```python
def node_f1(pred_tools, gold_tools):
    pred_set = set(pred_tools)
    gold_set = set(gold_tools)
    
    intersection = pred_set & gold_set
    precision = len(intersection) / len(pred_set) if pred_set else 0
    recall = len(intersection) / len(gold_set) if gold_set else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def normalized_edit_distance(pred_seq, gold_seq):
    distance = levenshtein(pred_seq, gold_seq)
    max_len = max(len(pred_seq), len(gold_seq))
    return distance / max_len if max_len > 0 else 0
```

**Category 3: Parameter Prediction**

Assesses the correctness of tool parameters at different granularity levels.

- **t-F1 (Tool-Parameter F1)**: F1 score for (tool, parameter_name) pairs
- **v-F1 (Tool-Parameter-Value F1)**: F1 score for (tool, parameter_name, parameter_value) triples

**Computation**:
```python
def t_f1(pred_steps, gold_steps):
    # Extract (tool, param_name) pairs
    pred_pairs = set()
    for step in pred_steps:
        tool = step['action']
        for param_name in step.get('arguments', {}).keys():
            pred_pairs.add((tool, param_name))
    
    gold_pairs = set()
    for step in gold_steps:
        tool = step['action']
        for param_name in step.get('arguments', {}).keys():
            gold_pairs.add((tool, param_name))
    
    return f1_over_sets(pred_pairs, gold_pairs)

def v_f1(pred_steps, gold_steps):
    # Extract (tool, param_name, param_value) triples
    pred_triples = set()
    for step in pred_steps:
        tool = step['action']
        for param_name, param_value in step.get('arguments', {}).items():
            pred_triples.add((tool, param_name, str(param_value)))
    
    gold_triples = set()
    for step in gold_steps:
        tool = step['action']
        for param_name, param_value in step.get('arguments', {}).items():
            gold_triples.add((tool, param_name, str(param_value)))
    
    return f1_over_sets(pred_triples, gold_triples)
```

#### Output Format

For each task, generates:

```json
{
  "id": 106,
  "text": "List all failure modes...",
  "task_decomposition": {
    "rouge1": 0.3704,
    "rouge2": 0.1923,
    "rougeL": 0.2963
  },
  "tool_selection": {
    "node_f1": 0.0,
    "edge_f1": 0.0,
    "node_set_accuracy": 0.0,
    "edge_set_accuracy": 0.0,
    "graph_accuracy": 0.0,
    "normalized_edit_distance": 1.0
  },
  "parameter_prediction": {
    "t_f1": 0.0,
    "v_f1": 0.0
  }
}
```

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Ground Truth Data                        │
│  (iot_gt.json, fsmr-gt.json)                                │
│  - Task questions                                            │
│  - Expected outputs                                          │
│  - Planning steps                                            │
│  - Execution steps & links                                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ├──────────────────┐
                     │                  │
                     ▼                  ▼
    ┌────────────────────────┐ ┌─────────────────────────┐
    │ Answer Quality         │ │ Agent Execution         │
    │ Question Generation    │ │ (CODS Track 2)          │
    │ (extract_complex_      │ │ - ReactAgent w/ Tools   │
    │  bench.py)             │ │ - Planning Workflow     │
    │                        │ │ - Enriched Trajectory   │
    └────────┬───────────────┘ └──────────┬──────────────┘
             │                            │
             │ Generates                  │ Generates
             ▼                            ▼
    ┌────────────────────────┐ ┌─────────────────────────┐
    │ Evaluation Questions   │ │ Agent Trajectory        │
    │ (complexbench_*_       │ │ (Q_*_trajectory.json)   │
    │  records.json)         │ │ - Question text         │
    │ - 6 quality dimensions │ │ - Planning steps        │
    │ - Few-shot examples    │ │ - Execution trace       │
    │ - Tool schemas         │ │ - Tool calls & args     │
    └────────┬───────────────┘ └──────────┬──────────────┘
             │                            │
             │                            │ Convert
             │                            ▼
             │                  ┌─────────────────────────┐
             │                  │ Evaluation Format       │
             │                  │ (predictions_*.json)    │
             │                  │ - Planning steps        │
             │                  │ - Execution steps       │
             │                  │ - Execution links       │
             │                  └──────────┬──────────────┘
             │                             │
             └─────────────┬───────────────┘
                           │
                           ▼
              ┌────────────────────────────┐
              │   Unified Evaluation       │
              │   (taskbench_eval.py)      │
              │                            │
              │   Answer Quality:          │
              │   - Helpfulness            │
              │   - Lexical correctness    │
              │   - Chain/Tool usage       │
              │   - Argument correctness   │
              │   - Factuality             │
              │   - Consistency            │
              │                            │
              │   Structural Correctness:  │
              │   - Task decomposition     │
              │   - Tool selection         │
              │   - Parameter prediction   │
              └────────────┬───────────────┘
                           │
                           ▼
              ┌────────────────────────────┐
              │   Evaluation Results       │
              │   (results_*.json)         │
              │   - Per-task scores        │
              │   - Aggregate metrics      │
              │   - Statistical summary    │
              └────────────────────────────┘
```

---

## Ground Truth Schema

### File Structure

Ground truth files (`iot_gt.json`, `fsmr-gt.json`) contain task specifications:

```json
[
  {
    "id": 106,
    "text": "List all failure modes of Chiller 6 at the MAIN site...",
    "expected_output": "The following failure modes can be detected:\n1. Chiller 6 Supply Temperature: High\n2. Chiller 6 Supply Temperature: Low",
    "planning_steps": [
      "Identify the site and asset",
      "Retrieve failure modes for Chiller 6",
      "Map failure modes to sensor"
    ],
    "execution_steps": [
      {
        "name": "get_failure_modes",
        "action": "Get Failure Modes",
        "arguments": "Chiller 6"
      },
      {
        "name": "get_mapping",
        "action": "Get Failure Mode and Sensor Relevancy Mapping",
        "arguments": "Chiller 6, <chiller_failure_list>, Chiller 6 Supply Temperature"
      },
      {
        "name": "finish",
        "action": "Finish",
        "arguments": ""
      }
    ],
    "execution_links": [
      {
        "source": "get_failure_modes",
        "target": "get_mapping"
      },
      {
        "source": "get_mapping",
        "target": "finish"
      }
    ],
    "characteristic_form": "Map(Asset, Sensor, FailureModes)"
  }
]
```

### Field Descriptions

| Field | Purpose | Used In |
|-------|---------|---------|
| `id` | Unique task identifier | Both evaluations |
| `text` | Task question | Both evaluations |
| `expected_output` | Gold-standard answer | Answer quality |
| `planning_steps` | Natural language sub-tasks | Task decomposition |
| `execution_steps` | Tool call specifications | Tool selection, parameters |
| `execution_links` | Step dependencies | Graph metrics |
| `characteristic_form` | Task category | Analysis |

---

## Trajectory Generation

### Step 1: Agent Execution

Agent trajectories are generated using the CODS Track 2 system with custom question support.

**Run with Ground Truth Questions**:

```bash
cd ../benchmark/cods_track2

# Start Docker container with fsmr-gt.json questions
docker compose up

# Or run specific tasks
docker compose run assetopsbench python /home/run_track_2.py \
  --utterance_ids 106,107,108 \
  --use_planning \
  --use_fsmr_gt
```

**Key Components**:

1. **ReactAgent**: Multi-tool reasoning agent
2. **PlanningDynamicWorkflow**: LLM-based task planning + dynamic execution
3. **Enriched Trajectory Tracking**: Captures planning, execution, and tool calls

**Output**: `track2_result/trajectory/Q_<id>_trajectory.json`

### Step 2: Trajectory Structure

Generated trajectories contain:

```json
{
  "id": 106,
  "text": "List all failure modes of Chiller 6...",
  "enriched_trajectory": {
    "workflow_type": "PlanningDynamicWorkflow",
    "total_runtime": 45.23,
    "llm_stats": {
      "total_calls": 12,
      "total_prompt_tokens": 8432,
      "total_completion_tokens": 1567
    },
    "execution_trace": [
      {
        "step_number": 1,
        "task_description": "Identify the site and asset",
        "agent_name": "FMSR Query Agent",
        "execution_details": {
          "react_iterations": [
            {
              "thought": "I need to use a failure mode tool...",
              "action": "Get Failure Modes",
              "action_input": {"asset_name": "Chiller 6"},
              "observation": "Found failure modes: [...",
              "llm_output": "..."
            }
          ],
          "final_answer": "..."
        }
      }
    ]
  }
}
```

**Critical Field**: `react_iterations` contains the actual tool calls made by the agent's reasoning process.

---

## Trajectory Conversion

### Converting to Evaluation Format

Trajectories must be converted to a standardized format for evaluation:

```bash
python3 convert_trajectories_corrected.py \
  ../benchmark/cods_track2/track2_result/trajectory \
  "Q_106_trajectory.json" \
  predictions_Q106.json
```

**Conversion Process**:

1. **Extract Planning Steps**: Combines task descriptions from execution trace
2. **Extract Execution Steps**: Parses `react_iterations` for tool calls
3. **Extract Execution Links**: Derives dependencies from sequential tool calls

**Example Conversion**:

```python
# Input: react_iterations
[
  {"action": "Get Failure Modes", "action_input": {"asset_name": "Chiller 6"}},
  {"action": "Get Mapping", "action_input": {"asset": "Chiller 6", "sensor": "..."}}
]

# Output: pred_execution_steps
[
  {"name": "step1", "action": "Get Failure Modes", "arguments": {"asset_name": "Chiller 6"}},
  {"name": "step2", "action": "Get Mapping", "arguments": {"asset": "Chiller 6", "sensor": "..."}},
  {"name": "finish", "action": "Finish", "arguments": {}}
]

# Output: pred_execution_links
[
  {"source": "step1", "target": "step2"},
  {"source": "step2", "target": "finish"}
]
```

### Output Format

```json
[
  {
    "id": 106,
    "pred_planning_steps": ["Identify the site and asset..."],
    "pred_execution_steps": [
      {"name": "step1", "action": "Get Failure Modes", "arguments": {...}},
      {"name": "step2", "action": "Get Mapping", "arguments": {...}}
    ],
    "pred_execution_links": [
      {"source": "step1", "target": "step2"}
    ]
  }
]
```

---

## Evaluation Execution

### Phase 1: Answer Quality Evaluation

**Generate Evaluation Questions**:

```bash
# For IoT tasks
python3 run_generation_incremental.py

# For FMSR tasks
python3 run_fmsr_generation.py
```

**Process**:
1. Loads ground truth from `iot_gt.json` or `fsmr-gt.json`
2. For each task, generates 6 evaluation questions (one per dimension)
3. Incorporates few-shot examples in prompts
4. Calls GPT-4 to generate evaluation criteria
5. Saves incrementally to `complexbench_*_records.json`

**Configuration**:
- Model: `gpt-4o-mini` or `gpt-4.1`
- Temperature: 0.0 (deterministic)
- Max tokens: 1500 per evaluation

**Output**: Evaluation questions ready for agent answer assessment

### Phase 2: Structural Correctness Evaluation

**Run Evaluation**:

```bash
python3 taskbench_eval.py \
  fsmr-gt.json \
  predictions_Q106.json \
  results_Q106.json
```

**Process**:
1. Loads ground truth planning, execution steps, and links
2. Loads predicted planning, execution steps, and links
3. Computes all metrics for each task:
   - ROUGE scores for planning
   - F1 and accuracy for tool selection
   - F1 for parameter prediction
4. Aggregates metrics across all tasks
5. Saves per-task and aggregate results

**Output**: Comprehensive evaluation results with per-task and aggregate scores

---

## Getting Started

### Prerequisites

```bash
# Python 3.8+
pip install openai nltk numpy

# For trajectory generation (Docker)
docker --version
```

### Quick Start

**1. Generate Answer Quality Evaluation Questions**:

```bash
cd /Users/ninad/Desktop/NYU/Dynamic_Evaluation/complex-bench-pipeline

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key"

# Generate for IoT
python3 run_generation_incremental.py

# Generate for FMSR
python3 run_fmsr_generation.py
```

**2. Generate Agent Trajectories**:

```bash
cd ../benchmark/cods_track2

# Run with ground truth questions
docker compose up
```

**3. Convert Trajectories**:

```bash
cd ../../complex-bench-pipeline

python3 convert_trajectories_corrected.py \
  ../benchmark/cods_track2/track2_result/trajectory \
  "Q_*_trajectory.json" \
  predictions_all.json
```

**4. Run Structural Evaluation**:

```bash
python3 taskbench_eval.py \
  fsmr-gt.json \
  predictions_all.json \
  results_structural.json
```

**5. View Results**:

```bash
cat results_structural.json | python3 -m json.tool
```

### File Organization

```
complex-bench-pipeline/
├── README.md                          # This file
├── extract_complex_bench.py           # Answer quality question generator
├── taskbench_eval.py                  # Structural correctness evaluator
├── convert_trajectories_corrected.py  # Trajectory converter
├── run_generation_incremental.py      # IoT evaluation generation
├── run_fmsr_generation.py             # FMSR evaluation generation
├── create_tool_schemas.py             # Tool schema extractor
├── iot_gt.json                        # IoT ground truth
├── fsmr-gt.json                       # FMSR ground truth
├── agent_tool_schemas.json            # IoT tool schemas
├── fmsr_tool_schemas.json             # FMSR tool schemas
├── complexbench_iot_records.json      # IoT evaluation questions
├── complexbench_fmsr_records.json     # FMSR evaluation questions
├── predictions_*.json                 # Converted trajectories
├── results_*.json                     # Evaluation results
└── .gitignore                         # Excludes large/sensitive files
```

---

## Results Interpretation

### Answer Quality Scores

**Helpfulness** (1-5):
- 5: Directly answers the question, actionable, comprehensive
- 3-4: Mostly helpful but may lack detail or completeness
- 1-2: Tangentially related or unhelpful

**Lexical Correctness** (Correct/Incorrect):
- Correct: All entity names, numbers, and terms are accurate
- Incorrect: Contains any lexical errors

**Chain & Tool Usage** (1-5):
- 5: Optimal tool sequence, efficient execution
- 3-4: Correct tools but suboptimal order
- 1-2: Wrong tools or illogical sequence

**Argument Correctness** (1-5):
- 5: All parameters are valid and appropriate
- 3-4: Minor parameter issues
- 1-2: Major parameter errors

**Factuality** (1-5):
- 5: All claims supported by evidence
- 3-4: Mostly factual with minor unsupported claims
- 1-2: Contains factual errors

**Consistency** (1-5):
- 5: Fully coherent and internally consistent
- 3-4: Minor inconsistencies
- 1-2: Major contradictions

### Structural Correctness Scores

**Task Decomposition** (ROUGE):
- > 0.5: Excellent planning alignment
- 0.3-0.5: Good planning similarity
- 0.1-0.3: Partial alignment
- < 0.1: Poor planning match

**Tool Selection** (F1/Accuracy):
- 1.0: Perfect tool selection
- 0.7-0.9: Mostly correct tools
- 0.3-0.7: Partial overlap
- < 0.3: Significant deviation

**Parameter Prediction** (t-F1, v-F1):
- 1.0: All parameters correct
- 0.7-0.9: Most parameters correct
- 0.3-0.7: Some parameters correct
- < 0.3: Few parameters correct

**Normalized Edit Distance** (NED):
- 0.0: Perfect tool sequence
- 0.0-0.3: Minor reordering
- 0.3-0.7: Significant differences
- 0.7-1.0: Completely different sequence

### Interpreting Low Scores

**Low Answer Quality, High Structural Correctness**:
- Agent executes correctly but produces poor final answers
- Possible issue: Answer generation or formatting

**High Answer Quality, Low Structural Correctness**:
- Agent finds alternative valid solutions
- Different approach than ground truth reference

**Both Low**:
- Fundamental agent execution issues
- Requires debugging

**Both High**:
- Excellent agent performance!

---

## Advanced Usage

### Custom Ground Truth

Create your own ground truth file:

```python
[
  {
    "id": 1,
    "text": "Your task question",
    "expected_output": "Expected answer",
    "planning_steps": ["Step 1", "Step 2"],
    "execution_steps": [
      {"name": "step1", "action": "ToolName", "arguments": "args"}
    ],
    "execution_links": [
      {"source": "step1", "target": "step2"}
    ]
  }
]
```

### Batch Evaluation

Evaluate multiple trajectories:

```bash
# Convert all trajectories
python3 convert_trajectories_corrected.py \
  ../benchmark/cods_track2/track2_result/trajectory \
  "Q_*_trajectory.json" \
  predictions_all_fsmr.json

# Evaluate
python3 taskbench_eval.py \
  fsmr-gt.json \
  predictions_all_fsmr.json \
  results_all_fsmr.json
```

### Custom Metrics

Extend `taskbench_eval.py` with your own metrics:

```python
def custom_metric(pred, gold):
    # Your metric logic
    return score

# Add to compute_task_metrics()
metrics["custom"] = custom_metric(pred_steps, gold_steps)
```

---

## Troubleshooting

### Trajectory Conversion Issues

**Problem**: No tool calls extracted

**Solution**: Verify `react_iterations` exists in trajectory:

```bash
cat Q_106_trajectory.json | jq '.enriched_trajectory.execution_trace[].execution_details.react_iterations'
```

### Evaluation Crashes

**Problem**: Missing ground truth task

**Solution**: Ensure prediction IDs match ground truth IDs

### Low Scores

**Problem**: All structural scores are 0%

**Solution**: Check if question text matches between trajectory and ground truth:

```bash
# Compare questions
python3 -c "
import json
traj = json.load(open('Q_106_trajectory.json'))
gt = json.load(open('fsmr-gt.json'))
print('Traj:', traj['text'])
print('GT:', [t['text'] for t in gt if t['id']==106][0])
"
```

---

## Citation

If you use this evaluation framework, please cite:

```bibtex
@software{multi_tool_agent_eval,
  title={Multi-Tool Agent Evaluation Framework},
  author={Your Name},
  year={2025},
  description={A comprehensive framework for evaluating multi-tool agents through answer quality and structural correctness}
}
```

---

## License

MIT License - see LICENSE file for details

---

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email].

---

**Happy Evaluating! 🎯**
