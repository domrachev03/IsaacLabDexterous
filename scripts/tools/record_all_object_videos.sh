#!/bin/bash
# Record inference video for each object using record_object_inference.py
#
# Usage:
#   ./scripts/tools/record_all_object_videos.sh <checkpoint_path> [output_dir] [num_envs] [video_length]
#
# Example:
#   ./scripts/tools/record_all_object_videos.sh logs/panda_rohand_reorient_pbt_agi.pth recordings 1 800

CHECKPOINT=${1:-"logs/panda_rohand_reorient_pbt_agi.pth"}
OUT_DIR=${2:-"recordings"}
NUM_ENVS=${3:-1}
VIDEO_LENGTH=${4:-800}
TASK="Isaac-Dexsuite-Panda-RoHand-Lift-Play-v0"

echo "==================================================="
echo "Recording object videos"
echo "Checkpoint: $CHECKPOINT"
echo "Output directory: $OUT_DIR"
echo "Num envs: $NUM_ENVS"
echo "Video length: $VIDEO_LENGTH"
echo "Task: $TASK"
echo "==================================================="

# Create output directory
mkdir -p "$OUT_DIR"

# Get the number of objects
echo "Getting object list..."
NUM_OBJECTS=$(python3 scripts/tools/eval_object_success.py \
    --task $TASK \
    --list-objects \
    --headless 2>/dev/null | grep "Total:" | awk '{print $2}')

if [ -z "$NUM_OBJECTS" ]; then
    echo "ERROR: Could not determine number of objects"
    exit 1
fi

echo "Found $NUM_OBJECTS objects to record"
echo ""

# Record each object directly to output folder (no subdirectories)
for i in $(seq 0 $((NUM_OBJECTS - 1))); do
    echo "=== Recording object $i/$((NUM_OBJECTS - 1)) ==="
    
    # Run recording - all videos go directly to OUT_DIR with custom names
    python3 scripts/tools/record_object_inference.py \
        --task $TASK \
        --checkpoint $CHECKPOINT \
        --object-index $i \
        --episodes 1 \
        --num-envs $NUM_ENVS \
        --video-folder "$OUT_DIR" \
        --video-length $VIDEO_LENGTH \
        --headless
    
    echo ""
done

echo "==================================================="
echo "All recordings complete!"
echo "Videos saved to: $OUT_DIR"
echo "==================================================="
