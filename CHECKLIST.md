# ✅ ComplexBench Pipeline - Quick Checklist

## 📋 Prerequisites

- [ ] You have Python 3.x installed
- [ ] You're in the correct directory: `complex-bench-pipeline/`

## 📁 Files Present (All ✓)

- [✓] `extract_complex_bench.py` - Main script with few-shot prompts
- [✓] `iot_gt.json` - Ground truth IoT tasks (101 tasks)
- [✓] `agent_tool_schemas.json` - Tool schemas (9 tools)
- [✓] `README.md` - Full documentation

## 🚀 Run Options

### Option A: Dry Run (No LLM - Just Prompts)

```bash
python3 extract_complex_bench.py
```

**Result:** Creates `complexbench_iot_records.json` with:
- ✓ All prompts filled in
- ✗ Annotations are `null` (no LLM calls)

**Status:** ✅ TESTED AND WORKING
- Generated 101 records
- Output file: 2.5 MB
- No errors

### Option B: With LLM Calls

**Steps:**

1. Install OpenAI SDK:
   ```bash
   pip install openai
   ```

2. Set API key:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

3. Update `call_llm` function in `extract_complex_bench.py` (see README.md)

4. Run:
   ```bash
   python3 extract_complex_bench.py
   ```

**Expected:**
- ~600-700 LLM calls (6-7 per task × 101 tasks)
- Time: 30-60 minutes
- Cost: Depends on model (gpt-4o vs gpt-4o-mini)

## 📊 Output Verification

Check the output:
```bash
# View file size
ls -lh complexbench_iot_records.json

# Count records
python3 -c "import json; print(len(json.load(open('complexbench_iot_records.json'))))"

# Inspect first record
python3 -c "import json; print(json.dumps(json.load(open('complexbench_iot_records.json'))[0], indent=2))" | head -50
```

## ✅ What You Have

| Item | Status | Notes |
|------|--------|-------|
| Ground truth file | ✅ | `iot_gt.json` (101 tasks) |
| Tool schemas | ✅ | `agent_tool_schemas.json` (9 tools) |
| Main script | ✅ | With few-shot prompts integrated |
| Output file | ✅ | Generated successfully (dry run) |
| Documentation | ✅ | README.md + README_FEWSHOT.md |

## 🎯 Few-Shot Prompts Included

All 6 metric types have few-shot examples:

1. **Helpfulness** - Examples: task id=1, id=5
2. **Lexical** - Examples: task id=3, id=9
3. **Chain/Tool** - Examples: task id=10 (multi-step)
4. **Arguments** - Examples: task id=9, id=1
5. **Factuality** - Examples: FMSR failure modes
6. **Consistency** - Examples: Contradiction detection

## 🔍 Sanity Checks

- [✓] No style/formatting/length questions
- [✓] Only "single" or "chain" composition
- [✓] All few-shots from real IoT tasks
- [✓] Metric-pure prompts (each focuses on one metric)
- [✓] Tool schemas cover all actions in iot_gt.json

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| `python: command not found` | Use `python3` instead |
| File not found | Make sure you're in `complex-bench-pipeline/` |
| JSON parse error | Check ground truth file format |
| Rate limiting | Use gpt-4o-mini or add delays |

## 📝 Next Steps

After successful run:

1. [ ] Inspect output: `head complexbench_iot_records.json`
2. [ ] Validate sample questions manually
3. [ ] Use dataset to evaluate agent responses
4. [ ] Calculate DRFR scores

---

**Current Status:** ✅ **READY TO USE**

The dry run completed successfully. You can:
- Use the prompts as-is (without LLM calls)
- Or implement LLM calls to generate actual evaluation questions

Run: `python3 extract_complex_bench.py`

