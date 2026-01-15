import os
import json
import sys

# Configuration
# We will append the session ID to this filename later
COUNTER_BASE_PATH = os.path.join(os.getcwd(), ".claude", "hooks")
CLAUDE_MD_PATH = "CLAUDE.md"
INTERVAL = 5

def main():
    # 1. Read the Session ID from Claude (passed via stdin)
    try:
        input_data = json.load(sys.stdin)
        # Fallback to "default" if running manually/testing without input
        session_id = input_data.get("session_id", "default") 
    except:
        session_id = "default"

    # 2. Create a unique counter file for this specific session
    counter_file = os.path.join(COUNTER_BASE_PATH, f"counter_{session_id}.txt")
    
    # 3. Increment the counter logic
    count = 0
    if os.path.exists(counter_file):
        try:
            with open(counter_file, 'r') as f:
                count = int(f.read().strip())
        except: pass
    
    count += 1
    os.makedirs(os.path.dirname(counter_file), exist_ok=True)
    with open(counter_file, 'w') as f:
        f.write(str(count))

    # 4. Check if we should remind
    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": "" 
        }
    }

    if count % INTERVAL == 0 and os.path.exists(CLAUDE_MD_PATH):
        with open(CLAUDE_MD_PATH, 'r') as f:
            content = f.read()
        
        output["hookSpecificOutput"]["additionalContext"] = (
            f"\n\n\n"
            f"--- AUTOMATED REMINDER (Message #{count}) ---\n"
            f"Here is the content of CLAUDE.md. You MUST adhere to these rules:\n"
            f"{content}\n"
            f"---------------------------------------------\n"
        )

    # 5. Print JSON to stdout
    print(json.dumps(output))

if __name__ == "__main__":
    main()
