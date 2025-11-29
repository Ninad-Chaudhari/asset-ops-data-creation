# Few-Shot Prompting Integration

## Overview

This document describes the few-shot prompting integration into the `extract_complex_bench.py` script.

## What Was Added

### 1. Six Few-Shot Constant Strings

At the top of the script (after imports), the following few-shot example strings were added:

- **FEWSHOT_HELPFULNESS**: Examples demonstrating helpfulness evaluation questions
  - Example 1: "What IoT sites are available?" (id=1)
  - Example 2: "Retrieve metadata for Chiller 6" (id=5)

- **FEWSHOT_LEXICAL**: Examples for lexical correctness evaluation
  - Example 1: "What assets can be found at MAIN site?" (id=3)
  - Example 2: "Retrieve sensor data for Chiller 6's % Loaded from June 2020" (id=9)

- **FEWSHOT_CHAIN_TOOL**: Examples for chain/tool usage evaluation
  - Example 1: "Get sensor data for both Chiller 6 and 9's Tonnage" (id=10)
  - Demonstrates multi-step chains with sensors, history, jsonfilemerge

- **FEWSHOT_ARGUMENTS**: Examples for argument correctness evaluation
  - Example 1: History call with all required parameters (id=9)
  - Example 2: Simple sites call with no parameters (id=1)

- **FEWSHOT_FACTUALITY**: Examples for factuality/supportiveness evaluation
  - Example 1: FMSR failure-mode list verification for Chiller 6

- **FEWSHOT_CONSISTENCY**: Examples for consistency evaluation
  - Example 1: Detecting contradictory statements about failure modes

### 2. Updated Prompt Builder Functions

Each of the six prompt builder functions was updated to include the corresponding few-shot examples:

1. `build_helpfulness_prompt()` → includes `{FEWSHOT_HELPFULNESS}`
2. `build_lexical_prompt()` → includes `{FEWSHOT_LEXICAL}`
3. `build_chain_and_tool_prompt()` → includes `{FEWSHOT_CHAIN_TOOL}`
4. `build_argument_prompt()` → includes `{FEWSHOT_ARGUMENTS}`
5. `build_factuality_prompt()` → includes `{FEWSHOT_FACTUALITY}`
6. `build_consistency_prompt()` → includes `{FEWSHOT_CONSISTENCY}`

### 3. Prompt Structure

Each updated prompt now follows this structure:

```
You are designing [METRIC] questions for ComplexBench-style evaluation...

{FEWSHOT_[METRIC]}

[CURRENT TASK]

[USER_INSTRUCTION]
{instruction}

[CHARACTERISTIC_FORM]
{char_form}

[FINAL_ANSWER]
{final_answer}

Goal: Generate YES/NO questions that test [METRIC]...
```

The `[CURRENT TASK]` section clearly separates the few-shot examples from the actual task being evaluated.

## Benefits

1. **Better LLM Guidance**: The few-shot examples provide concrete demonstrations of:
   - What kinds of questions to generate
   - How to structure question IDs and dependencies
   - The expected JSON output format
   - The appropriate granularity and specificity

2. **Consistency**: By showing examples from the ground truth IoT dataset, the LLM better understands the domain-specific requirements

3. **Quality Control**: Examples demonstrate what NOT to ask about (e.g., style, formatting, JSON vs Markdown)

## Usage

The script remains backwards compatible. Simply call the functions as before:

```python
from extract_complex_bench import convert_all_tasks_to_complexbench

convert_all_tasks_to_complexbench(
    "iot_gt.json", 
    "agent_tool_schemas.json",
    "complexbench_iot_records.json"
)
```

The few-shot examples will automatically be included in all LLM prompts.

## Optional Enhancements

For future consideration:

1. **Context Management**: If prompts become too long, consider:
   - Randomly sampling 1-2 examples per run instead of showing all
   - Truncating examples to save tokens
   
2. **Dynamic Selection**: Select the most relevant few-shot examples based on task characteristics (e.g., single vs chain composition)

3. **Additional Examples**: Add more few-shot examples for edge cases or complex scenarios as they are identified

## Notes

- No changes to the core logic or data flow
- All linter checks pass
- The `build_constraint_and_composition_prompt()` function was NOT updated as constraint/composition few-shots were marked as optional

