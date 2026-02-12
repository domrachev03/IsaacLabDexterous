#!/bin/bash
# Run eval_object_success.py for all objects and collect results
#
# Usage:
#   ./scripts/tools/run_all_object_eval.sh <checkpoint_path> [episodes_per_object] [num_envs] [enable_cameras]
#
# Example:
#   ./scripts/tools/run_all_object_eval.sh logs/panda_rohand_reorient_pbt_agi.pth 20 8
#   ./scripts/tools/run_all_object_eval.sh logs/panda_rohand_reorient_pbt_agi.pth 20 8 true

CHECKPOINT=${1:-"logs/panda_rohand_reorient_pbt_agi.pth"}
EPISODES=${2:-20}
NUM_ENVS=${3:-8}
ENABLE_CAMERAS=${4:-"false"}
TASK="Isaac-Dexsuite-Panda-RoHand-Lift-Play-v0"

# Prepare camera flag
CAMERA_FLAG=""
if [ "$ENABLE_CAMERAS" = "true" ] || [ "$ENABLE_CAMERAS" = "1" ]; then
    CAMERA_FLAG="--enable_cameras"
    TASK="Isaac-Dexsuite-UR10-Tessolo-Lift-Visible-Play-v0"
fi

# Output file for results
RESULTS_FILE="eval_results_$(date +%Y%m%d_%H%M%S).txt"

echo "==================================================="
echo "Running object evaluation"
echo "Checkpoint: $CHECKPOINT"
echo "Episodes per object: $EPISODES"
echo "Num envs: $NUM_ENVS"
echo "Enable cameras: $ENABLE_CAMERAS"
echo "Task: $TASK"
echo "Results file: $RESULTS_FILE"
echo "==================================================="

# First, get the number of objects
echo "Getting object list..."
NUM_OBJECTS=$(python3 scripts/tools/eval_object_success.py \
    --task $TASK \
    --list-objects \
    --headless \
    $CAMERA_FLAG 2>/dev/null | grep "Total:" | awk '{print $2}')

if [ -z "$NUM_OBJECTS" ]; then
    echo "ERROR: Could not determine number of objects"
    exit 1
fi

echo "Found $NUM_OBJECTS objects to evaluate"
echo ""

# Header for results file
echo "Object Evaluation Results" > $RESULTS_FILE
echo "Checkpoint: $CHECKPOINT" >> $RESULTS_FILE
echo "Episodes: $EPISODES" >> $RESULTS_FILE
echo "Date: $(date)" >> $RESULTS_FILE
echo "==================================================" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# Initialize totals
TOTAL_SUCCESSES=0
TOTAL_EPISODES=0

# Arrays to store results for final table
declare -a RESULT_INDICES
declare -a RESULT_NAMES
declare -a RESULT_SUCCS
declare -a RESULT_TOTS
declare -a RESULT_RATES

# Evaluate each object
for i in $(seq 0 $((NUM_OBJECTS - 1))); do
    echo "=== Evaluating object $i/$((NUM_OBJECTS - 1)) ==="
    
    # Run evaluation and capture output
    OUTPUT=$(python3 scripts/tools/eval_object_success.py \
        --task $TASK \
        --checkpoint $CHECKPOINT \
        --object-index $i \
        --episodes $EPISODES \
        --num-envs $NUM_ENVS \
        --headless 2>&1 \
        $CAMERA_FLAG)
    
    # Extract parseable result
    RESULT=$(echo "$OUTPUT" | grep "PARSEABLE_RESULT" | tail -1)
    
    if [ -n "$RESULT" ]; then
        # Parse: PARSEABLE_RESULT:index:name:successes:total:rate
        INDEX=$(echo $RESULT | cut -d: -f2)
        NAME=$(echo $RESULT | cut -d: -f3)
        SUCC=$(echo $RESULT | cut -d: -f4)
        TOT=$(echo $RESULT | cut -d: -f5)
        RATE=$(echo $RESULT | cut -d: -f6)
        
        echo "$INDEX: $NAME - $SUCC/$TOT = $(echo "scale=2; $RATE * 100" | bc)%"
        echo "$INDEX,$NAME,$SUCC,$TOT,$RATE" >> $RESULTS_FILE
        
        # Store for final table
        RESULT_INDICES+=("$INDEX")
        RESULT_NAMES+=("$NAME")
        RESULT_SUCCS+=("$SUCC")
        RESULT_TOTS+=("$TOT")
        RESULT_RATES+=("$RATE")
        
        TOTAL_SUCCESSES=$((TOTAL_SUCCESSES + SUCC))
        TOTAL_EPISODES=$((TOTAL_EPISODES + TOT))
    else
        echo "ERROR: Failed to evaluate object $i"
        echo "$i,ERROR,0,0,0" >> $RESULTS_FILE
        
        # Store error for final table
        RESULT_INDICES+=("$i")
        RESULT_NAMES+=("ERROR")
        RESULT_SUCCS+=("0")
        RESULT_TOTS+=("0")
        RESULT_RATES+=("0")
    fi
    
    echo ""
done

# Calculate overall stats
if [ $TOTAL_EPISODES -gt 0 ]; then
    OVERALL_RATE=$(echo "scale=4; $TOTAL_SUCCESSES / $TOTAL_EPISODES" | bc)
else
    OVERALL_RATE=0
fi

# Print final results table
echo ""
echo "==========================================================================="
echo "                        EVALUATION RESULTS TABLE"
echo "==========================================================================="
echo "Checkpoint: $CHECKPOINT"
echo "Episodes per object: $EPISODES"
echo "---------------------------------------------------------------------------"
printf "| %-5s | %-40s | %-8s | %-10s |\n" "Index" "Object Type" "Success" "Rate"
echo "---------------------------------------------------------------------------"

for j in $(seq 0 $((${#RESULT_INDICES[@]} - 1))); do
    RATE_PCT=$(echo "scale=1; ${RESULT_RATES[$j]} * 100" | bc)
    printf "| %-5s | %-40s | %3s/%-4s | %6s%% |\n" \
        "${RESULT_INDICES[$j]}" \
        "${RESULT_NAMES[$j]:0:40}" \
        "${RESULT_SUCCS[$j]}" \
        "${RESULT_TOTS[$j]}" \
        "$RATE_PCT"
done

echo "---------------------------------------------------------------------------"
OVERALL_PCT=$(echo "scale=1; $OVERALL_RATE * 100" | bc)
printf "| %-5s | %-40s | %3s/%-4s | %6s%% |\n" \
    "ALL" "TOTAL" "$TOTAL_SUCCESSES" "$TOTAL_EPISODES" "$OVERALL_PCT"
echo "==========================================================================="
echo ""
echo "Results saved to: $RESULTS_FILE"

# Add summary to results file
echo "" >> $RESULTS_FILE
echo "==================================================" >> $RESULTS_FILE
echo "TOTAL,$TOTAL_SUCCESSES,$TOTAL_EPISODES,$OVERALL_RATE" >> $RESULTS_FILE
