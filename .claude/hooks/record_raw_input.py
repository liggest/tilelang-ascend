import json
import sys
import os

try:
    input_data = json.load(sys.stdin)
except json.JSONDecodeError as e:
    print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
    sys.exit(1)

CLAUDE_PROJECT_DIR = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
LOG_FILE = os.path.join(CLAUDE_PROJECT_DIR, "logs", "session_input.jsonl")

try:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        json.dump(input_data, f, ensure_ascii=False)
        f.write("\n")
except IOError as e:
    print(f"Error: Failed to write to log file: {e}", file=sys.stderr)
    sys.exit(1)