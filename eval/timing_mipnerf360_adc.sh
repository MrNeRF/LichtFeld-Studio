#!/bin/bash

SCENE_DIR="data"
RESULT_DIR="results/benchmark"
SCENE_LIST="garden bicycle stump bonsai counter kitchen room" # treehill flowers

# Start total timer
total_start=$(date +%s)

# Initialize array to store individual times
declare -A scene_times

for SCENE in $SCENE_LIST;
do
    # Determine data factor based on scene type
    if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
        DATA_FACTOR=2
    else
        DATA_FACTOR=4
    fi
    
    echo "========================================="
    echo "Running $SCENE with images_${DATA_FACTOR}"
    echo "========================================="
    
    # Start timer for this scene
    scene_start=$(date +%s)
    
    # Run training with evaluation
    ./build/LichtFeld-Studio \
        -d $SCENE_DIR/$SCENE/ \
        -o $RESULT_DIR/$SCENE/ \
        --images images_${DATA_FACTOR} \
        --iter 30000 \
        --headless \
		--config eval/adc_optimization_params.json

    # End timer for this scene
    scene_end=$(date +%s)
    scene_duration=$((scene_end - scene_start))
    scene_times[$SCENE]=$scene_duration
    
    # Format time nicely
    hours=$((scene_duration / 3600))
    minutes=$(((scene_duration % 3600) / 60))
    seconds=$((scene_duration % 60))
    
    echo "Completed $SCENE in ${hours}h ${minutes}m ${seconds}s"
    echo
done

# End total timer
total_end=$(date +%s)
total_duration=$((total_end - total_start))

# Print summary
echo "========================================="
echo "BENCHMARK SUMMARY"
echo "========================================="

# Print individual scene times
for SCENE in $SCENE_LIST; do
    duration=${scene_times[$SCENE]}
    hours=$((duration / 3600))
    minutes=$(((duration % 3600) / 60))
    seconds=$((duration % 60))
    printf "%-15s: %02dh %02dm %02ds\n" "$SCENE" "$hours" "$minutes" "$seconds"
done

echo "-----------------------------------------"

# Print total time
total_hours=$((total_duration / 3600))
total_minutes=$(((total_duration % 3600) / 60))
total_seconds=$((total_duration % 60))
echo "Total time: ${total_hours}h ${total_minutes}m ${total_seconds}s"
echo "========================================="
