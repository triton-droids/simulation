#!/bin/bash
export PYTHONPATH=$(pwd)

# Define training script path
TRAIN_SCRIPT="src/scripts/train.py"

# Define task and environment parameters
TASK="locomotion"
ENV="humanoid_legs"
TERRAIN="flat"

# Video recording parameters
VIDEO=true
VIDEO_LENGTH=1000
VIDEO_INTERVAL=50000000

LOG_PROJECT_NAME="default_humanoid_legs_locomotion"

python $TRAIN_SCRIPT \
    --video_length="$VIDEO_LENGTH" \
    --video_interval="$VIDEO_INTERVAL" \
    --task="$TASK" \
    --log_project_name="$LOG_PROJECT_NAME" \
    --video
