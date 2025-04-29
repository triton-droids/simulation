#!/bin/bash
export PYTHONPATH=$(pwd)

# Define training script path
TRAIN_SCRIPT="src/scripts/train.py"

# Define task and environment parameters
TASK="locomotion"
ENV="humanoid_legs"
TERRAIN="flat"

# Video recording parameters
VIDEO_LENGTH=1000
VIDEO_INTERVAL=50000000

LOG_PROJECT_NAME="random_sweep_humanoid"

# Declare associative arrays for parameters
declare -A PARAMS

# Function to define a parameter and its values (as space-separated string)
add_param() {
    local name=$1
    shift
    PARAMS[$name]="$*"
}

# Function to select a random element from a space-separated string
select_random() {
    local values=($1)
    echo "${values[RANDOM % ${#values[@]}]}"
}

# Define parameter values
add_param "LIN_VEL"            "1.0 1.5 2.0"
add_param "ANG_VEL"            "0.5 1.0"
add_param "ANG_VEL_XY"         "-0.3 -0.15 -0.05"
add_param "BASE_HEIGHT"        "0.0 -0.5 -1.0"
add_param "ORIENTATION"        "-2.0 -1.0 -0.5"
add_param "FEET_SLIP"          "-0.5 -0.25 -0.1"
add_param "FEET_PHASE"         "1.0 2.0 3.0"
add_param "POSE"               "-1.0 -0.5 -0.1"
add_param "SURVIVAL"           "0.0 -1.0 -5.0"
add_param "LEARNING_RATE"      "1e-3 3e-4 1e-4"
add_param "CLIPPING_EPSILON"   "0.1 0.2 0.3"
add_param "TRACKING_SIGMA"     "1.0 10.0 50.0 100.0"
# Run 200 randomized sweeps sequentially
for i in {1..36}; do
    # Randomize parameters
    LIN_VEL=$(select_random "${PARAMS[LIN_VEL]}")
    ANG_VEL=$(select_random "${PARAMS[ANG_VEL]}")
    ANG_VEL_XY=$(select_random "${PARAMS[ANG_VEL_XY]}")
    BASE_HEIGHT=$(select_random "${PARAMS[BASE_HEIGHT]}")
    ORIENTATION=$(select_random "${PARAMS[ORIENTATION]}")
    FEET_SLIP=$(select_random "${PARAMS[FEET_SLIP]}")
    FEET_PHASE=$(select_random "${PARAMS[FEET_PHASE]}")
    POSE=$(select_random "${PARAMS[POSE]}")
    SURVIVAL=$(select_random "${PARAMS[SURVIVAL]}")
    LEARNING_RATE=$(select_random "${PARAMS[LEARNING_RATE]}")
    CLIPPING_EPSILON=$(select_random "${PARAMS[CLIPPING_EPSILON]}")
    TRACKING_SIGMA=$(select_random "${PARAMS[TRACKING_SIGMA]}")

    echo "Running sweep $i with random parameters..."

    python $TRAIN_SCRIPT \
        --video_length="$VIDEO_LENGTH" \
        --video_interval="$VIDEO_INTERVAL" \
        --task="$TASK" \
        --log_project_name="$LOG_PROJECT_NAME" \
        --video \
        agent.learning_rate="$LEARNING_RATE" \
        agent.entropy_cost="$CLIPPING_EPSILON" \
        sim.rewards.tracking_sigma="$TRACKING_SIGMA"
        # sim.reward_scales.lin_vel="$LIN_VEL" \
        # sim.reward_scales.ang_vel="$ANG_VEL" \
        # sim.reward_scales.ang_vel_xy="$ANG_VEL_XY" \
        # sim.reward_scales.orientation="$ORIENTATION" \
        # sim.reward_scales.feet_slip="$FEET_SLIP" \
        # sim.reward_scales.feet_phase="$FEET_PHASE" \
        # sim.reward_scales.pose="$POSE" \
        # sim.reward_scales.survival="$SURVIVAL" 
        # sim.reward_scales.survival="$SURVIVAL" \
         # sim.reward_scales.base_height="$BASE_HEIGHT" \

    # Wait for the current run to finish before starting the next one
    wait
done
