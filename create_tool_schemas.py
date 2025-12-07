"""
Helper script to create agent_tool_schemas.json from IoTAgent tools.
This extracts the tool schemas needed by extract_complex_bench.py.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path to import IoTAgent
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir / "src"))

try:
    from IoTAgent.gettools import getTools
except ImportError as e:
    print(f"Error importing IoTAgent: {e}")
    print("Creating minimal schema file instead...")
    # Create a minimal schema file based on what we see in iot_gt.json
    minimal_schemas = [
        {
            "name": "sites",
            "description": "List all available IoT sites",
            "parameters": {"type": "object", "properties": {}},
            "Agent": "IoTAgent",
            "Agent_Description": "IoT data agent",
            "Sample_Examples": ["Get all sites"]
        },
        {
            "name": "assets",
            "description": "List assets at a specific site",
            "parameters": {
                "type": "object",
                "properties": {
                    "site_name": {"type": "string", "description": "Name of the site"}
                },
                "required": ["site_name"]
            },
            "Agent": "IoTAgent",
            "Agent_Description": "IoT data agent",
            "Sample_Examples": ["List assets at MAIN site"]
        },
        {
            "name": "sensors",
            "description": "List sensors for a specific asset at a site",
            "parameters": {
                "type": "object",
                "properties": {
                    "site_name": {"type": "string", "description": "Name of the site"},
                    "assetnum": {"type": "string", "description": "Asset identifier"}
                },
                "required": ["site_name", "assetnum"]
            },
            "Agent": "IoTAgent",
            "Agent_Description": "IoT data agent",
            "Sample_Examples": ["Get sensors for Chiller 6 at MAIN"]
        },
        {
            "name": "history",
            "description": "Retrieve historical sensor data",
            "parameters": {
                "type": "object",
                "properties": {
                    "site_name": {"type": "string", "description": "Name of the site"},
                    "assetnum": {"type": "string", "description": "Asset identifier"},
                    "sensor_name": {"type": "string", "description": "Name of the sensor"},
                    "start": {"type": "string", "description": "Start timestamp (ISO format)"},
                    "final": {"type": "string", "description": "End timestamp (ISO format)"}
                },
                "required": ["site_name", "assetnum", "sensor_name", "start", "final"]
            },
            "Agent": "IoTAgent",
            "Agent_Description": "IoT data agent",
            "Sample_Examples": ["Get Chiller 6 temperature history from June 2020"]
        },
        {
            "name": "jsonreader",
            "description": "Read and parse JSON files",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "Name of the JSON file to read"}
                },
                "required": ["file_name"]
            },
            "Agent": "IoTAgent",
            "Agent_Description": "IoT data agent",
            "Sample_Examples": ["Read data from file.json"]
        },
        {
            "name": "jsonfilemerge",
            "description": "Merge two JSON files into one",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name_1": {"type": "string", "description": "First JSON file"},
                    "file_name_2": {"type": "string", "description": "Second JSON file"}
                },
                "required": ["file_name_1", "file_name_2"]
            },
            "Agent": "IoTAgent",
            "Agent_Description": "IoT data agent",
            "Sample_Examples": ["Merge file1.json and file2.json"]
        },
        {
            "name": "currenttime",
            "description": "Get the current time",
            "parameters": {"type": "object", "properties": {}},
            "Agent": "IoTAgent",
            "Agent_Description": "IoT data agent",
            "Sample_Examples": ["What is the current time?"]
        },
        {
            "name": "Finish",
            "description": "Finish the task and return the final answer",
            "parameters": {
                "type": "object",
                "properties": {
                    "argument": {"type": "string", "description": "Final answer to return"}
                },
                "required": ["argument"]
            },
            "Agent": "IoTAgent",
            "Agent_Description": "IoT data agent",
            "Sample_Examples": []
        }
    ]
    
    output_path = Path(__file__).parent / "agent_tool_schemas.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(minimal_schemas, f, indent=2, ensure_ascii=False)
    print(f"✓ Created minimal agent_tool_schemas.json at {output_path}")
    sys.exit(0)


def extract_tool_schemas():
    """Extract tool schemas from IoTAgent tools."""
    tools = getTools()
    schemas = []
    
    for tool in tools:
        # Try to extract schema information from the tool
        schema = {
            "name": getattr(tool, "name", tool.__class__.__name__),
            "description": getattr(tool, "description", ""),
            "parameters": getattr(tool, "parameters", {}),
            "Agent": "IoTAgent",
            "Agent_Description": "IoT data agent for querying building management systems",
            "Sample_Examples": getattr(tool, "examples", [])
        }
        schemas.append(schema)
    
    return schemas


if __name__ == "__main__":
    schemas = extract_tool_schemas()
    
    # Write to file
    output_path = Path(__file__).parent / "agent_tool_schemas.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(schemas, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Created agent_tool_schemas.json with {len(schemas)} tool schemas")
    print(f"✓ Saved to: {output_path}")


