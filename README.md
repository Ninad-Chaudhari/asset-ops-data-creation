# ComplexBench Dataset Builder - Quick Start Guide

This folder contains everything you need to convert IoT ground truth data into ComplexBench-style evaluation datasets.

## 📁 Required Files (Already Present!)

Your directory structure is ready to go:

```
complex-bench-pipeline/
├── extract_complex_bench.py       ← Main script (with few-shot prompts)
├── iot_gt.json                    ← Ground truth IoT tasks ✓
├── agent_tool_schemas.json        ← Tool schemas ✓
└── README.md                      ← This file
```

✅ All required files are present!

## 🚀 How to Run

### Option 1: Dry Run (No LLM calls - Just Generate Prompts)

This will create the output file with all prompts but no LLM-generated questions:

```bash
cd complex-bench-pipeline
python3 extract_complex_bench.py
```

**Output:** `complexbench_iot_records.json` with:
- ✓ All prompts filled in under `complexbench_metadata`
- ✗ All `complexbench_annotations` fields will be `null`

This is useful to:
- Inspect the prompts before making LLM calls
- Verify the pipeline works
- Check that few-shot examples are properly included

### Option 2: With LLM Calls (Generate Actual Questions)

To actually generate evaluation questions, you need to implement the `call_llm` function.

#### Step 1: Install OpenAI SDK

```bash
pip install openai
```

#### Step 2: Set your API key

```bash
export OPENAI_API_KEY="sk-..."
```

#### Step 3: Update the `call_llm` function

Open `extract_complex_bench.py` and replace the stub function (lines 8-20) with:

```python
from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from environment

def call_llm(prompt: str, model: str = "gpt-4o") -> Optional[Dict[str, Any]]:
    """
    Call OpenAI API to generate evaluation questions.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        
        # Extract the text response
        text = response.choices[0].message.content
        
        # The prompt instructs the model to return JSON
        # Strip any markdown code fences if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        # Parse and return JSON
        return json.loads(text)
    except Exception as e:
        print(f"⚠️  LLM call failed: {e}")
        return None
```

#### Step 4: Run the script

```bash
python3 extract_complex_bench.py
```

This will:
1. Process each task in `iot_gt.json`
2. Generate 6-7 prompts per task (with few-shot examples)
3. Call the LLM for each prompt
4. Save results to `complexbench_iot_records.json`

**Note:** This will make ~6-7 LLM calls per task. With 142+ tasks in `iot_gt.json`, expect:
- **~850-1000 LLM calls total**
- **Time:** ~30-60 minutes (depending on rate limits)
- **Cost:** Depends on your model choice (gpt-4o, gpt-4o-mini, etc.)

## 📊 Output Format

The output file `complexbench_iot_records.json` will contain an array of records:

```json
[
  {
    "id": "c50d83ca-6082-447d-a030-70b38a4e4a4b",
    "original_task": {
      "text": "What IoT sites are available?",
      "characteristic_form": "first call action sites with no parameters",
      "category": "Knowledge Query",
      "composition_hint": "single"
    },
    "complexbench_metadata": {
      "constraint_and_composition_prompt": "...",
      "helpfulness_prompt": "...",
      "lexical_prompt": "...",
      "factuality_prompt": "...",
      "chain_tool_prompt": "...",
      "argument_prompt": "...",
      "consistency_prompt": "..."
    },
    "complexbench_annotations": {
      "constraint_and_composition": {...},
      "helpfulness_questions": {
        "questions": [
          {
            "id": "H1",
            "text": "Does the answer clearly state all available IoT sites?",
            "type": "helpfulness",
            "depends_on": []
          }
        ]
      },
      "lexical_questions": {...},
      "factuality_questions": {...},
      "chain_tool_questions": {...},
      "argument_questions": {...},
      "consistency_questions": {...}
    }
  },
  ...
]
```

## 🎯 What the Script Does

### 1. Few-Shot Prompting
Each prompt includes 1-2 examples from real IoT tasks demonstrating:
- Expected question format
- Proper metric focus (no style/formatting questions)
- Dependencies structure
- JSON output format

### 2. Metrics Covered
The script generates evaluation questions for 6 core metrics:

| Metric | What It Checks |
|--------|---------------|
| **Helpfulness** | Does the answer solve the user's task? |
| **Lexical** | Are asset/site/sensor/time names correct? |
| **Factuality** | Does the answer match ground truth data? |
| **Chain/Tool** | Is the tool sequence sensible? |
| **Arguments** | Are tool parameters correct? |
| **Consistency** | Is the answer internally consistent? |

### 3. Composition Types
Tasks are classified as:
- `single`: One atomic subtask
- `chain`: Multiple sequential subtasks

(We do NOT use 'And', 'Selection', or 'Nested' for this dataset)

## ✅ Sanity Checks

Before running at scale, verify:

1. **No forbidden metrics**: Script does NOT ask about JSON formatting, bullet points, length, persona, sentiment
2. **Composition limited**: Only "single" or "chain" (no Selection/And/Nested)
3. **Few-shots aligned**: All examples come from real IoT tasks in `iot_gt.json`
4. **Metric purity**: Each prompt focuses on ONE metric type

## 🔧 Troubleshooting

### "File not found" errors
Make sure you're in the correct directory:
```bash
cd /Users/ninad/Desktop/NYU/Dynamic_Evaluation/complex-bench-pipeline
```

### "ModuleNotFoundError: No module named 'openai'"
Install the OpenAI SDK:
```bash
pip install openai
```

### Rate limiting errors
If you hit rate limits:
1. Use a cheaper/faster model like `gpt-4o-mini`
2. Add delays between calls
3. Process in batches

### JSON parsing errors from LLM
The LLM might occasionally return invalid JSON. The script handles this gracefully by:
- Returning `null` for that annotation
- Printing a warning
- Continuing with other tasks

## 📈 Next Steps

After generating the dataset:

1. **Inspect** the output:
   ```bash
   head -n 100 complexbench_iot_records.json | jq
   ```

2. **Validate** the questions manually sample a few records

3. **Use** the dataset to evaluate agent responses

4. **Calculate** DRFR scores (Dependency-Aware Recall and Failure Rate)

## 🤝 Support

If you encounter issues:
1. Check this README
2. Verify file structure matches above
3. Run in dry-run mode first
4. Check the few-shot examples are loading correctly

---

**Ready to go!** Just run `python3 extract_complex_bench.py` 🚀

