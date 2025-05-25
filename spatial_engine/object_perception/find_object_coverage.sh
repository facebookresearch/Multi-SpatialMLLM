# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash
# run_tasks.sh
# Usage: ./run_tasks.sh [start] [chunk_size]
# This script runs object coverage finding for both train and val splits

# Set default values
DEFAULT_START=0
DEFAULT_CHUNK_SIZE=10

# Get command line parameters or use default values
START=${1:-$DEFAULT_START}
CHUNK_SIZE=${2:-$DEFAULT_CHUNK_SIZE}

echo "Launching tasks for both train and val splits from scene $START with chunk size $CHUNK_SIZE"

# Process train split
echo "Processing train split..."
for ((current_start=START; ; current_start+=CHUNK_SIZE)); do
    current_end=$((current_start+CHUNK_SIZE))
    echo "  Starting task for train: scenes $current_start to $current_end"
    # Launch task in background, output files will include start and end suffixes
    python spatial_engine/object_perception/single_object_coverage_finder.py --split "train" --start "$current_start" --end "$current_end" &
    
    # Add a limit to prevent infinite loop - can be adjusted as needed
    if [ $current_start -ge 1000 ]; then
        break
    fi
done

# Process val split
echo "Processing val split..."
for ((current_start=START; ; current_start+=CHUNK_SIZE)); do
    current_end=$((current_start+CHUNK_SIZE))
    echo "  Starting task for val: scenes $current_start to $current_end"
    # Launch task in background, output files will include start and end suffixes
    python spatial_engine/object_perception/single_object_coverage_finder.py --split "val" --start "$current_start" --end "$current_end" &
    
    # Add a limit to prevent infinite loop - can be adjusted as needed
    if [ $current_start -ge 1000 ]; then
        break
    fi
done

# Wait for all background tasks to complete
wait
echo "All tasks completed for both train and val splits."