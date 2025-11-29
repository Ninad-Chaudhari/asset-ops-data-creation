import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============== FEW-SHOT EXAMPLES ===============

FEWSHOT_HELPFULNESS = r"""
[FEW-SHOT EXAMPLES]

Example 1
[USER_INSTRUCTION]
What IoT sites are available?

[CHARACTERISTIC_FORM]
first call action sites with no parameters

[FINAL_ANSWER]
The available IoT site is MAIN.  The final answer is The available IoT site is MAIN.

[QUESTIONS_JSON]
{
  "questions": [
    {
      "id": "H1",
      "text": "Does the answer clearly state all available IoT sites (e.g., MAIN) requested by the user?",
      "type": "helpfulness",
      "depends_on": []
    }
  ]
}

---

Example 2
[USER_INSTRUCTION]
Retrieve metadata for Chiller 6 located at the MAIN site.

[CHARACTERISTIC_FORM]
The expected response should be the metadata for asset 'Chiller 6' at the MAIN site. The metadata may be in the return value, or may be returned as a reference to a file containing the metadata.

[FINAL_ANSWER]
The metadata for Chiller 6 at MAIN site includes: Chiller 6 Condenser Water Return To Tower Temperature, Chiller 6 Chiller Efficiency, Chiller 6 Tonnage, Chiller 6 Supply Temperature, Chiller 6 Return Temperature, Chiller 6 Run Status, Chiller 6 Condenser Water Flow, Chiller 6 Schedule, Chiller 6 Power Input, Chiller 6 Chiller % Loaded, Chiller 6 Liquid Refrigerant Evaporator Temperature, Chiller 6 Setpoint Temperature.

[QUESTIONS_JSON]
{
  "questions": [
    {
      "id": "H1",
      "text": "Does the answer actually provide the metadata for asset 'Chiller 6' at the MAIN site as requested?",
      "type": "helpfulness",
      "depends_on": []
    }
  ]
}
"""

FEWSHOT_LEXICAL = r"""
[FEW-SHOT EXAMPLES]

Example 1
[USER_INSTRUCTION]
What assets can be found at the MAIN site?

[CHARACTERISTIC_FORM]
The expected response should be the return value from querying the assets at the MAIN site. The response should be a reference to a file containing the list of assets.

[FINAL_ANSWER]
The assets at the MAIN site are: CQPA AHU 1, CQPA AHU 2B, Chiller 4, Chiller 6, Chiller 9, Chiller 3.

[QUESTIONS_JSON]
{
  "questions": [
    {
      "id": "L1",
      "text": "Does the answer correctly use the site name 'MAIN' as specified in the instruction?",
      "type": "lexical",
      "depends_on": []
    },
    {
      "id": "L2",
      "text": "Does the answer list asset names (e.g., CQPA AHU 1, Chiller 6) with correct spelling and formatting?",
      "type": "lexical",
      "depends_on": []
    }
  ]
}

---

Example 2
[USER_INSTRUCTION]
Retrieve sensor data for Chiller 6's % Loaded from June 2020 at MAIN.

[CHARACTERISTIC_FORM]
The expected response should be the % Loaded measurements for the asset 'Chiller 6' at the MAIN site. It should include all recorded values for June 1, 2020 through June 30, 2020 inclusive. The response should be a reference to a file containing the % Loaded data.

[FINAL_ANSWER]
The sensor data for Chiller 6's % Loaded at the MAIN site from June 1, 2020 through June 30, 2020 has been saved to uuid.json.

[QUESTIONS_JSON]
{
  "questions": [
    {
      "id": "L1",
      "text": "Does the answer correctly mention asset 'Chiller 6' and site 'MAIN'?",
      "type": "lexical",
      "depends_on": []
    },
    {
      "id": "L2",
      "text": "Does the answer clearly indicate that the % Loaded measurements come from the full period June 1, 2020 through June 30, 2020 inclusive?",
      "type": "lexical",
      "depends_on": []
    },
    {
      "id": "L3",
      "text": "Does the answer correctly reference a file (e.g., uuid.json) instead of inlining the raw time-series values, as required?",
      "type": "lexical",
      "depends_on": []
    }
  ]
}
"""

FEWSHOT_CHAIN_TOOL = r"""
[FEW-SHOT EXAMPLES]

Example 1
[USER_INSTRUCTION]
Get sensor data for both Chiller 6 and Chiller 9's Tonnage from first week of June 2020 at MAIN in a single file.

[CHARACTERISTIC_FORM]
The expected response should be the Tonnage values for both asset 'Chiller 6' and asset 'Chiller 9' at the MAIN site for last week. The values from both chillers should be merged into a single file. The return value should reference the file containing the merged Tonnage data from both chillers.

[EXECUTION_STEPS]
- step 'step1': action='sensors', agent='IoTAgent', arguments={"site_name": "MAIN", "assetnum": "Chiller 6"}
- step 'step2': action='sensors', agent='IoTAgent', arguments={"site_name": "MAIN", "assetnum": "Chiller 9"}
- step 'step3': action='jsonreader', agent='IoTAgent', arguments={"file_name": "uuid.json"}
- step 'step4': action='jsonreader', agent='IoTAgent', arguments={"file_name": "uuid.json"}
- step 'step5': action='history', agent='IoTAgent', arguments={"site_name": "MAIN", "assetnum": "Chiller 6", "start": "2020-06-01T00:00:00.000000+00:00", "final": "2020-06-07T00:00:00.000000+00:00"}
- step 'step6': action='history', agent='IoTAgent', arguments={"site_name": "MAIN", "assetnum": "Chiller 9", "start": "2020-06-01T00:00:00.000000+00:00", "final": "2020-06-07T00:00:00.000000+00:00"}
- step 'step7': action='jsonfilemerge', agent='IoTAgent', arguments={"file_name_1": "file1.json", "file_name_2": "file2.json"}
- step 'finish': action='Finish', agent='IoTAgent', argument="uuid.json"

[QUESTIONS_JSON]
{
  "questions": [
    {
      "id": "C1",
      "text": "Does the chain first query sensors for both Chiller 6 and Chiller 9 before fetching history data?",
      "type": "chain_tool_correctness",
      "depends_on": []
    },
    {
      "id": "C2",
      "text": "Does the chain call history separately for Chiller 6 and Chiller 9 over the correct date range?",
      "type": "chain_tool_correctness",
      "depends_on": []
    },
    {
      "id": "C3",
      "text": "Does the chain merge the two time-series files into a single file using jsonfilemerge before finishing?",
      "type": "chain_tool_correctness",
      "depends_on": ["C1", "C2"]
    }
  ]
}
"""

FEWSHOT_ARGUMENTS = r"""
[FEW-SHOT EXAMPLES]

Example 1
[USER_INSTRUCTION]
Retrieve sensor data for Chiller 6's % Loaded from June 2020 at MAIN.

[EXECUTION_STEPS]
- step 'step1': action='sensors', arguments={"site_name": "MAIN", "assetnum": "Chiller 6"}
- step 'step2': action='jsonreader', arguments={"file_name": "uuid.json"}
- step 'step3': action='history', arguments={
    "site_name": "MAIN",
    "assetnum": "Chiller 6",
    "start": "2020-06-01T00:00:00.000000+00:00",
    "final": "2020-06-30T23:59:59.000000+00:00",
    "sensor_name": "Chiller 6 Chiller % Loaded"
  }

[RELEVANT_TOOL_SCHEMAS]
history requires parameters: site_name, assetnum, start, final, sensor_name

[QUESTIONS_JSON]
{
  "questions": [
    {
      "id": "A1",
      "text": "Does the history call include all required parameters (site_name, assetnum, start, final, sensor_name)?",
      "type": "argument_correctness",
      "depends_on": []
    },
    {
      "id": "A2",
      "text": "Are the start and final timestamps in the history call consistent with the requested June 1–30, 2020 range?",
      "type": "argument_correctness",
      "depends_on": []
    },
    {
      "id": "A3",
      "text": "Is the sensor_name argument exactly 'Chiller 6 Chiller % Loaded' as implied by the instruction?",
      "type": "argument_correctness",
      "depends_on": []
    }
  ]
}

---

Example 2
[USER_INSTRUCTION]
What IoT sites are available?

[EXECUTION_STEPS]
- step 'step1': action='sites', arguments={{}}
- step 'finish': action='Finish', argument="The available IoT site is MAIN.  The final answer is The available IoT site is MAIN."

[RELEVANT_TOOL_SCHEMAS]
sites requires no parameters.

[QUESTIONS_JSON]
{
  "questions": [
    {
      "id": "A1",
      "text": "Is the sites tool called with an empty argument object, as required (no extra unused parameters)?",
      "type": "argument_correctness",
      "depends_on": []
    }
  ]
}
"""

FEWSHOT_FACTUALITY = r"""
[FEW-SHOT EXAMPLES]

Example 1
[USER_INSTRUCTION]
List all failure modes of asset Chiller 6 at the MAIN site.

[CHARACTERISTIC_FORM]
the answer should contain one or more failure modes of Chiller 6. The failure modes of Chiller 6 need to be from the list ['Compressor Overheating: Failed due to Normal wear, overheating', 'Heat Exchangers: Fans: Degraded motor or worn bearing due to Normal use', 'Evaporator Water side fouling', 'Condenser Water side fouling', 'Condenser Improper water side flow rate', 'Purge Unit Excessive purge', 'Refrigerant Operated Control Valve Failed spring'].

[OBSERVED_GROUND_TRUTH_FROM_TOOLS]
The failure modes of asset Chiller 6: ['Compressor Overheating: Failed due to Normal wear, overheating', 'Heat Exchangers: Fans: Degraded motor or worn bearing due to Normal use', 'Evaporator Water side fouling', 'Condenser Water side fouling', 'Condenser Improper water side flow rate', 'Purge Unit Excessive purge', 'Refrigerant Operated Control Valve Failed spring'].

[FINAL_ANSWER]
The failure modes of Chiller 6 at the MAIN site are: Compressor Overheating: Failed due to Normal wear, overheating; Heat Exchangers: Fans: Degraded motor or worn bearing due to Normal use; Evaporator Water side fouling; Condenser Water side fouling; Condenser Improper water side flow rate; Purge Unit Excessive purge; Refrigerant Operated Control Valve Failed spring.

[QUESTIONS_JSON]
{
  "questions": [
    {
      "id": "F1",
      "text": "Does the answer list only failure modes that appear in the ground truth list for Chiller 6?",
      "type": "factuality_supportiveness",
      "depends_on": []
    },
    {
      "id": "F2",
      "text": "Does the answer avoid omitting any failure modes that are present in the ground truth list?",
      "type": "factuality_supportiveness",
      "depends_on": []
    }
  ]
}
"""

FEWSHOT_CONSISTENCY = r"""
[FEW-SHOT EXAMPLES]

Example 1
[USER_INSTRUCTION]
List all failure modes of asset Chiller 6 at the MAIN site.

[OBSERVED_GROUND_TRUTH_FROM_TOOLS]
The failure modes of asset Chiller 6: ['Compressor Overheating: Failed due to Normal wear, overheating', 'Heat Exchangers: Fans: Degraded motor or worn bearing due to Normal use', 'Evaporator Water side fouling', 'Condenser Water side fouling', 'Condenser Improper water side flow rate', 'Purge Unit Excessive purge', 'Refrigerant Operated Control Valve Failed spring'].

[FINAL_ANSWER]
The failure modes of Chiller 6 at the MAIN site are: Compressor Overheating: Failed due to Normal wear, overheating; Heat Exchangers: Fans: Degraded motor or worn bearing due to Normal use; Evaporator Water side fouling. Other failure modes do not exist.

[QUESTIONS_JSON]
{
  "questions": [
    {
      "id": "K1",
      "text": "Does the answer avoid asserting that certain listed ground-truth failure modes 'do not exist' when they actually do exist in the ground truth?",
      "type": "consistency",
      "depends_on": []
    },
    {
      "id": "K2",
      "text": "Is the answer internally consistent about which failure modes exist for Chiller 6?",
      "type": "consistency",
      "depends_on": []
    }
  ]
}
"""


# =============== LLM CALL ===============

def call_llm(prompt: str, model: str = "gpt-4o-mini") -> Optional[Dict[str, Any]]:
    """
    Call OpenAI API to generate evaluation questions.
    
    Returns parsed JSON from the model response, or None if call fails.
    """
    try:
        from openai import OpenAI
        import os
        
        # Get API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at creating evaluation questions for agent benchmarks. Always respond with valid JSON only, no additional commentary."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=2000,
        )
        
        # Extract the text response
        text = response.choices[0].message.content.strip()
        
        # Strip markdown code fences if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        # Parse and return JSON
        return json.loads(text)
        
    except Exception as e:
        print(f"⚠️  LLM call failed: {e}")
        return None


# =============== LOADING & UTILITIES ===============

def load_json(path: str) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def index_tool_schemas(tool_schema_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Build a mapping from tool name -> schema object.
    """
    result = {}
    for tool in tool_schema_list:
        name = tool.get("name")
        if name:
            result[name] = tool
    return result


def get_finish_argument(task: Dict[str, Any]) -> str:
    """
    Get the final answer text for IoT tasks.

    We prefer:
    - execution_steps[name=='finish'].argument (IoTAgent traces)
    - else: final_out_description joined
    - else: empty string.
    """
    # 1) Look in execution_steps for Finish
    for step in task.get("execution_steps", []):
        if step.get("action") == "Finish":
            arg = step.get("argument")
            if isinstance(arg, str):
                return arg

    # 2) Some IoT tasks have final_out_description
    fod = task.get("final_out_description")
    if isinstance(fod, list):
        return " ".join(str(x) for x in fod)

    # 3) Fallback: nothing
    return ""


def infer_composition_type(task: Dict[str, Any]) -> str:
    """
    Infer composition type: 'single' or 'chain' from execution_steps + execution_links.

    - If only 0 or 1 meaningful execution steps before Finish -> 'single'
    - Else -> 'chain'

    We *do not* try to infer And/Selection/Nested because the trace is linear.
    """
    steps = task.get("execution_steps", [])
    if not steps:
        return "single"

    non_finish_steps = [s for s in steps if s.get("action") != "Finish"]
    if len(non_finish_steps) <= 1:
        return "single"
    return "chain"


def summarize_execution_steps(task: Dict[str, Any]) -> str:
    """
    Turn execution_steps into a compact text summary we can safely hand to the LLM
    without blowing context.

    We include: name, action, agent, and arguments.
    """
    lines = []
    for step in task.get("execution_steps", []):
        name = step.get("name", "")
        action = step.get("action", "")
        agent = step.get("agent", "")
        args = step.get("arguments", {})
        lines.append(
            f"- step '{name}': action='{action}', agent='{agent}', arguments={json.dumps(args, ensure_ascii=False)}"
        )
    return "\n".join(lines)


def summarize_relevant_tool_schemas(
    task: Dict[str, Any],
    tool_schemas_by_name: Dict[str, Dict[str, Any]],
    max_tools: int = 6,
) -> str:
    """
    Take only the schemas for tools that appear in this task's execution_steps,
    to keep context under control.
    """
    used_actions = {s.get("action") for s in task.get("execution_steps", []) if s.get("action")}
    chunks: List[str] = []
    for action in sorted(used_actions):
        if action not in tool_schemas_by_name:
            continue
        schema = tool_schemas_by_name[action]
        desc = schema.get("description", "")
        params = schema.get("parameters", {})
        agent = schema.get("Agent", "")
        agent_desc = schema.get("Agent_Description", "")
        examples = schema.get("Sample_Examples", [])[:3]

        chunk = [
            f"Tool name: {action}",
            f"  Description: {desc}",
            f"  Agent: {agent}",
            f"  Agent_Description: {agent_desc}",
            f"  Parameters JSON schema: {json.dumps(params, ensure_ascii=False)}",
        ]
        if examples:
            ex_str = "; ".join(examples)
            chunk.append(f"  Sample_Examples: {ex_str}")
        chunks.append("\n".join(chunk))
        if len(chunks) >= max_tools:
            break

    if not chunks:
        return ""
    return "\n\n".join(chunks)


# =============== PROMPT BUILDERS (ONE PER METRIC GROUP) ===============

def build_constraint_and_composition_prompt(task: Dict[str, Any]) -> str:
    """
    Ask the LLM to label which constraint *dimensions* are present in the instruction,
    and to confirm composition type (single/chain).
    """
    instruction = task.get("text", "")
    char_form = task.get("characteristic_form", "")
    composition_hint = infer_composition_type(task)
    exec_summary = summarize_execution_steps(task)

    return f"""
You are annotating an instruction for ComplexBench-style evaluation.

[USER_INSTRUCTION]
{instruction}

[CHARACTERISTIC_FORM]
{char_form}

[EXECUTION_STEPS_SUMMARY]
{exec_summary}

Your tasks:

1. Identify which of the following constraint dimensions are **explicitly present**
   in the instruction and characteristic_form (do not infer ones that are not mentioned):
   - lexical: asset/site/sensor/time/file naming correctness
   - utility_helpfulness
   - utility_factuality_supportiveness
   - utility_consistency
   (Do NOT mark format/length/json/markdown/bullets/persona/sentiment/style/target-language.)

2. Confirm the composition type as one of:
   - "single"  (there is effectively one atomic subtask)
   - "chain"   (there are multiple subtasks that must be completed in sequence)
   (We do NOT use 'And', 'Selection', or 'Nested' for this dataset.)

Output JSON:

{{
  "constraint_dimensions": [
    "lexical",
    "utility_helpfulness",
    "utility_factuality_supportiveness",
    "utility_consistency"
  ],
  "composition_type": "{composition_hint}"
}}

Only include entries that truly apply; omit any that do not.
"""


def build_helpfulness_prompt(task: Dict[str, Any]) -> str:
    instruction = task.get("text", "")
    char_form = task.get("characteristic_form", "")
    final_answer = get_finish_argument(task)

    return f"""
You are designing evaluation questions for a ComplexBench-style benchmark
for an IoT multi-tool agent.

{FEWSHOT_HELPFULNESS}

[CURRENT TASK]

[USER_INSTRUCTION]
{instruction}

[CHARACTERISTIC_FORM]
{char_form}

[FINAL_ANSWER]
{final_answer}

Goal: Generate YES/NO questions that test only HELPFULNESS:
- Helpfulness = whether the final answer actually solves the user's task as
  described in the instruction and characteristic_form, regardless of style or formatting.

Rules:
- Do NOT ask about JSON vs Markdown vs bullets.
- Do NOT ask about output length.
- Do NOT ask about tone, sentiment, or persona.
- Do NOT ask about the internal tool sequence (that is a separate metric).

For each question, provide:
- "id": like "H1", "H2", ...
- "text": the yes/no question
- "type": "helpfulness"
- "depends_on": a list of other question ids (empty list if none)

Output JSON:
{{ "questions": [{{"id":"H1","text":"...","type":"helpfulness","depends_on":[]}}, ...] }}
"""


def build_lexical_prompt(task: Dict[str, Any]) -> str:
    instruction = task.get("text", "")
    char_form = task.get("characteristic_form", "")
    final_answer = get_finish_argument(task)

    return f"""
You are designing LEXICAL correctness questions for ComplexBench-style evaluation
on an IoT multi-tool task.

{FEWSHOT_LEXICAL}

[CURRENT TASK]

[USER_INSTRUCTION]
{instruction}

[CHARACTERISTIC_FORM]
{char_form}

[FINAL_ANSWER]
{final_answer}

Lexical correctness here means:
- Correct asset names (e.g., Chilllers, AHUs),
- Correct site names,
- Correct sensor names,
- Correct time range mentions (when explicitly requested),
- Whether the answer correctly returns a FILE REFERENCE vs inline content
  if that is specified in the characteristic_form.

Rules:
- Do NOT ask about style, JSON layout, or length.
- Do NOT ask about argument correctness of tool calls explicitly (that's another metric),
  but you CAN ask whether user-facing entities (assets/sites/sensors/times/files) are correct.

For each question, provide:
- "id": like "L1", "L2"
- "text": the yes/no question
- "type": "lexical"
- "depends_on": a list of other question ids (may be empty)

Output JSON:
{{ "questions": [{{"id":"L1","text":"...","type":"lexical","depends_on":[]}}, ...] }}
"""


def build_factuality_prompt(task: Dict[str, Any]) -> str:
    instruction = task.get("text", "")
    char_form = task.get("characteristic_form", "")
    final_answer = get_finish_argument(task)
    # We add a short summary of observable ground truth (final_out / final_out_description)
    final_out = task.get("final_out")
    final_out_desc = task.get("final_out_description", [])
    final_out_str = json.dumps(final_out, ensure_ascii=False) if final_out else ""
    final_out_desc_str = " ".join(str(x) for x in final_out_desc)

    return f"""
You are designing FACTUALITY/SUPPORTIVENESS questions for ComplexBench-style evaluation
on an IoT benchmark.

{FEWSHOT_FACTUALITY}

[CURRENT TASK]

[USER_INSTRUCTION]
{instruction}

[CHARACTERISTIC_FORM]
{char_form}

[OBSERVED_GROUND_TRUTH_FROM_TOOLS]
final_out: {final_out_str}
final_out_description: {final_out_desc_str}

[FINAL_ANSWER]
{final_answer}

Factuality/supportiveness here means:
- The final answer should not contradict the ground truth information in final_out
  and final_out_description when those are non-empty.
- It should not claim data exists when ground truth explicitly says it is missing, and vice versa.
- It should faithfully summarize counts, categories, or recommendations when those are given.

Rules:
- Do NOT hallucinate hidden databases beyond what is shown in the ground truth.
- Do NOT ask about style or formatting.
- Focus only on whether statements in the answer align with the above ground truth.

For each question, provide:
- "id": like "F1", "F2"
- "text": the yes/no question
- "type": "factuality_supportiveness"
- "depends_on": a list of other question ids (may be empty)

Output JSON:
{{ "questions": [{{"id":"F1","text":"...","type":"factuality_supportiveness","depends_on":[]}}, ...] }}
"""


def build_chain_and_tool_prompt(
    task: Dict[str, Any], tool_schemas_by_name: Dict[str, Dict[str, Any]]
) -> str:
    instruction = task.get("text", "")
    char_form = task.get("characteristic_form", "")
    exec_summary = summarize_execution_steps(task)
    schema_snippets = summarize_relevant_tool_schemas(task, tool_schemas_by_name)
    composition_hint = infer_composition_type(task)

    return f"""
You are designing CHAIN/TOOL-USAGE correctness questions for ComplexBench-style evaluation
for an IoT agent.

{FEWSHOT_CHAIN_TOOL}

[CURRENT TASK]

[USER_INSTRUCTION]
{instruction}

[CHARACTERISTIC_FORM]
{char_form}

[EXECUTION_STEPS]
{exec_summary}

[RELEVANT_TOOL_SCHEMAS]
{schema_snippets}

Composition hint for this task: "{composition_hint}" (either "single" or "chain").

We want YES/NO questions that check:
- Whether the tools used and their sequence (chain) are sensible for solving the task.
- Whether important intermediate steps (e.g., getting sensors before history, merging files, etc.)
  were performed.
- Whether unnecessary or obviously wrong tools were used.

Rules:
- Do NOT ask about the user-facing answer content (that is covered by other metrics).
- Do NOT ask about formatting, style, or length.
- Focus on internal tool sequence and choice.

For each question, provide:
- "id": like "C1", "C2"
- "text": the yes/no question
- "type": "chain_tool_correctness"
- "depends_on": a list of other question ids (may be empty)

Output JSON:
{{ "questions": [{{"id":"C1","text":"...","type":"chain_tool_correctness","depends_on":[]}}, ...] }}
"""


def build_argument_prompt(
    task: Dict[str, Any], tool_schemas_by_name: Dict[str, Dict[str, Any]]
) -> str:
    instruction = task.get("text", "")
    exec_summary = summarize_execution_steps(task)
    schema_snippets = summarize_relevant_tool_schemas(task, tool_schemas_by_name)

    return f"""
You are designing ARGUMENT correctness questions for ComplexBench-style evaluation.

{FEWSHOT_ARGUMENTS}

[CURRENT TASK]

[USER_INSTRUCTION]
{instruction}

[EXECUTION_STEPS]
{exec_summary}

[RELEVANT_TOOL_SCHEMAS]
{schema_snippets}

Argument correctness here means:
- For each tool call, all required parameters defined in its JSON schema are present.
- The parameter values (site_name, assetnum, start, final, etc.) match the user instruction
  and any constraints in the characteristic_form.
- Parameter types are correct (string vs null, etc.).

Rules:
- Do NOT ask about final answer wording (handled in other metrics).
- Do NOT ask about style or format.
- Focus only on arguments passed to tools, using the provided schemas.

For each question, provide:
- "id": like "A1", "A2"
- "text": the yes/no question
- "type": "argument_correctness"
- "depends_on": a list of other question ids (may be empty)

Output JSON:
{{ "questions": [{{"id":"A1","text":"...","type":"argument_correctness","depends_on":[]}}, ...] }}
"""


def build_consistency_prompt(task: Dict[str, Any]) -> str:
    instruction = task.get("text", "")
    char_form = task.get("characteristic_form", "")
    final_answer = get_finish_argument(task)
    final_out = task.get("final_out")
    final_out_desc = task.get("final_out_description", [])
    final_out_str = json.dumps(final_out, ensure_ascii=False) if final_out else ""
    final_out_desc_str = " ".join(str(x) for x in final_out_desc)

    return f"""
You are designing CONSISTENCY questions for ComplexBench-style evaluation.

{FEWSHOT_CONSISTENCY}

[CURRENT TASK]

[USER_INSTRUCTION]
{instruction}

[CHARACTERISTIC_FORM]
{char_form}

[OBSERVED_GROUND_TRUTH_FROM_TOOLS]
final_out: {final_out_str}
final_out_description: {final_out_desc_str}

[FINAL_ANSWER]
{final_answer}

Consistency here means:
- The answer does not contradict itself.
- The answer does not contradict the explicit ground truth from final_out/final_out_description.
- The answer does not both claim that data exists and that it does not exist for the same item.

Rules:
- Do NOT ask about tool sequences (covered by chain_tool_correctness).
- Do NOT ask about style, formatting, or length.

For each question, provide:
- "id": like "K1", "K2"
- "text": the yes/no question
- "type": "consistency"
- "depends_on": a list of other question ids (may be empty)

Output JSON:
{{ "questions": [{{"id":"K1","text":"...","type":"consistency","depends_on":[]}}, ...] }}
"""


# =============== MAIN CONVERSION PIPELINE ===============

def build_complexbench_record_for_task(
    task: Dict[str, Any],
    tool_schemas_by_name: Dict[str, Dict[str, Any]],
    call_llm_fn=call_llm,
) -> Dict[str, Any]:
    """
    Convert a single IoT ground-truth task into a ComplexBench-style record.

    We:
    - Keep raw fields (instruction, characteristic_form, etc.)
    - Generate prompts (and optionally LLM outputs) for each metric group
    - Attach them to the record.
    """
    record_id = task.get("uuid") or task.get("id")
    instruction = task.get("text", "")
    char_form = task.get("characteristic_form", "")
    category = task.get("category", "")
    composition = infer_composition_type(task)

    # Build prompts
    constraint_prompt = build_constraint_and_composition_prompt(task)
    helpfulness_prompt = build_helpfulness_prompt(task)
    lexical_prompt = build_lexical_prompt(task)
    factuality_prompt = build_factuality_prompt(task)
    chain_prompt = build_chain_and_tool_prompt(task, tool_schemas_by_name)
    argument_prompt = build_argument_prompt(task, tool_schemas_by_name)
    consistency_prompt = build_consistency_prompt(task)

    # Optionally call LLM here (currently returns None to keep pure)
    constraint_annotations = call_llm_fn(constraint_prompt)  # expected JSON or None
    helpfulness_questions = call_llm_fn(helpfulness_prompt)
    lexical_questions = call_llm_fn(lexical_prompt)
    factuality_questions = call_llm_fn(factuality_prompt)
    chain_questions = call_llm_fn(chain_prompt)
    argument_questions = call_llm_fn(argument_prompt)
    consistency_questions = call_llm_fn(consistency_prompt)

    # Assemble record
    record: Dict[str, Any] = {
        "id": record_id,
        "original_task": {
            "text": instruction,
            "characteristic_form": char_form,
            "category": category,
            "composition_hint": composition,
        },
        "complexbench_metadata": {
            # These store prompts so you can re-run or audit them later
            "constraint_and_composition_prompt": constraint_prompt,
            "helpfulness_prompt": helpfulness_prompt,
            "lexical_prompt": lexical_prompt,
            "factuality_prompt": factuality_prompt,
            "chain_tool_prompt": chain_prompt,
            "argument_prompt": argument_prompt,
            "consistency_prompt": consistency_prompt,
        },
        "complexbench_annotations": {
            # LLM outputs (if call_llm is wired up); otherwise None placeholders.
            "constraint_and_composition": constraint_annotations,
            "helpfulness_questions": helpfulness_questions,
            "lexical_questions": lexical_questions,
            "factuality_questions": factuality_questions,
            "chain_tool_questions": chain_questions,
            "argument_questions": argument_questions,
            "consistency_questions": consistency_questions,
        },
    }

    return record


def convert_all_tasks_to_complexbench(
    gt_path: str,
    schemas_path: str,
    out_path: str,
    call_llm_fn=call_llm,
) -> None:
    """
    Main entry: go through all ground-truth tasks and produce a JSON list of ComplexBench-style
    records.

    It is *critical* that we do not hallucinate unsupported metrics (format/semantic/selection/etc.),
    so this function only builds the subset we know is grounded.
    """
    gt_data = load_json(gt_path)
    tool_schema_list = load_json(schemas_path)
    tool_schemas_by_name = index_tool_schemas(tool_schema_list)

    # iot_gt.json appears to be a list of tasks
    if not isinstance(gt_data, list):
        raise ValueError("Expected iot_gt.json to be a list of task objects")

    records: List[Dict[str, Any]] = []

    for task in gt_data:
        # Only process tasks that have an instruction text (skip low-level entries)
        if not isinstance(task, dict) or "text" not in task:
            continue

        record = build_complexbench_record_for_task(
            task,
            tool_schemas_by_name,
            call_llm_fn=call_llm_fn,
        )
        records.append(record)

    out_p = Path(out_path)
    with out_p.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(records)} ComplexBench-style records to {out_p}")


if __name__ == "__main__":
    # Run the conversion pipeline
    convert_all_tasks_to_complexbench(
        gt_path="iot_gt.json",
        schemas_path="agent_tool_schemas.json",
        out_path="complexbench_iot_records.json",
    )
