#!/bin/bash
# Quick monitoring script for parallel contextual NN jobs
# Usage: ./monitor_parallel.sh [log_dir]

LOG_DIR=${1:-$(ls -td logs/parallel_contextual_nn_* 2>/dev/null | head -1)}

if [ -z "$LOG_DIR" ] || [ ! -d "$LOG_DIR" ]; then
    echo "No log directory found. Usage: $0 <log_dir>"
    exit 1
fi

echo "Monitoring: $LOG_DIR"
echo "========================================"

for log in "$LOG_DIR"/*.log; do
    name=$(basename "$log" .log)
    
    # Check if process is still running (look for recent writes)
    if [ -f "$log" ]; then
        # Get last line for status
        last_line=$(tail -1 "$log" 2>/dev/null)
        
        # Count completed visual layers (look for "Saved to" messages)
        completed=$(grep -c "✓ Saved to" "$log" 2>/dev/null || echo 0)
        
        # Check for errors
        errors=$(grep -ci "error\|exception\|traceback" "$log" 2>/dev/null || echo 0)
        
        # Check if done
        if grep -q "✓ DONE!" "$log" 2>/dev/null; then
            status="✓ DONE"
        elif [ "$errors" -gt 0 ]; then
            status="✗ ERROR ($errors)"
        else
            status="⏳ Running"
        fi
        
        printf "%-25s %s  [%d layers done]\n" "$name:" "$status" "$completed"
    fi
done

echo "========================================"
echo ""
echo "Tail latest activity:"
tail -1 "$LOG_DIR"/*.log 2>/dev/null | head -20

