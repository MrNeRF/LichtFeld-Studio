#!/bin/bash

SCENE_DIR="data"
RESULT_DIR="results/mrnf"
STRATEGY_NAME="MRNF"
SCENE_LIST="garden bicycle stump bonsai counter kitchen room" # treehill flowers

# Check if results directory exists and prompt for deletion
if [ -d "$RESULT_DIR" ]; then
    echo "Results directory '$RESULT_DIR' already exists."
    read -p "Do you want to delete it and start fresh? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing results directory..."
        rm -rf "$RESULT_DIR"
    else
        echo "Keeping existing results. New results will overwrite existing ones for each scene."
    fi
    echo
fi

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

    # Run training with evaluation, capturing wall-clock duration.
    mkdir -p "$RESULT_DIR/$SCENE"
    scene_start=$(date +%s.%N)
    ./build/LichtFeld-Studio \
        -d $SCENE_DIR/$SCENE/ \
        -o $RESULT_DIR/$SCENE/ \
        --images images_${DATA_FACTOR} \
        --test-every 8 \
        --eval \
        --headless \
        --save-eval-images \
        --config eval/mrnf_optimization_params.json
    scene_end=$(date +%s.%N)
    scene_elapsed=$(echo "$scene_end - $scene_start" | bc -l)
    printf "%.2f\n" "$scene_elapsed" > "$RESULT_DIR/$SCENE/training_time_seconds.txt"

    echo "Completed $SCENE (training: ${scene_elapsed}s)"
    echo
done

# Function to format numbers to specified decimal places
format_number() {
    local num=$1
    local decimals=$2
    printf "%.${decimals}f" $num
}

# Function to format numbers with thousands separators
format_with_commas() {
    local num=$1
    echo $num | sed ':a;s/\B[0-9]\{3\}\>/,&/;ta'
}

# Format seconds as either "Xs" or "MmSs" (e.g. "212.34s" or "3m32.34s")
format_duration() {
    local total=$1
    if [ -z "$total" ] || [ "$total" = "0" ]; then
        echo "n/a"
        return
    fi
    local minutes=$(echo "$total / 60" | bc)
    local secs=$(echo "$total - $minutes * 60" | bc -l)
    if [ "$minutes" -gt 0 ]; then
        printf "%dm%05.2fs" "$minutes" "$secs"
    else
        printf "%.2fs" "$total"
    fi
}

# Print formatted results table
echo
echo "================================================================================"
echo "QUALITY METRICS SUMMARY $STRATEGY_NAME"
echo "================================================================================"
printf "%-10s %-10s %-10s %-10s %-12s %-15s\n" "scene" "iteration" "psnr" "ssim" "time" "num_gaussians"
echo "--------------------------------------------------------------------------------"

# Collect and format results for each scene
total_psnr=0
total_ssim=0
total_gaussians=0
total_time=0
valid_scenes=0

for SCENE in $SCENE_LIST;
do
    csv_file="$RESULT_DIR/$SCENE/metrics.csv"
    if [ -f "$csv_file" ]; then
        # Get the last line of metrics (final iteration)
        final_metrics=$(tail -n 1 "$csv_file")

        # Parse CSV values (format: iteration,psnr,ssim,time_per_image,num_gaussians)
        IFS=',' read -r iteration psnr ssim time_per_image num_gaussians <<< "$final_metrics"

        # Read training wall-clock time for this scene
        time_file="$RESULT_DIR/$SCENE/training_time_seconds.txt"
        if [ -f "$time_file" ]; then
            scene_time=$(cat "$time_file")
        else
            scene_time=0
        fi

        # Format the numbers
        psnr_fmt=$(format_number $psnr 4)
        ssim_fmt=$(format_number $ssim 6)
        gaussians_fmt=$(format_with_commas $num_gaussians)
        time_fmt=$(format_duration "$scene_time")

        # Print formatted row
        printf "%-10s %-10s %-10s %-10s %-12s %-15s\n" \
            "$SCENE" \
            "$iteration" \
            "$psnr_fmt" \
            "$ssim_fmt" \
            "$time_fmt" \
            "$gaussians_fmt"

        echo "--------------------------------------------------------------------------------"

        # Accumulate for mean calculation
        total_psnr=$(echo "$total_psnr + $psnr" | bc -l)
        total_ssim=$(echo "$total_ssim + $ssim" | bc -l)
        total_gaussians=$((total_gaussians + num_gaussians))
        total_time=$(echo "$total_time + $scene_time" | bc -l)
        valid_scenes=$((valid_scenes + 1))
    fi
done

# Calculate and print mean
if [ $valid_scenes -gt 0 ]; then
    mean_psnr=$(echo "$total_psnr / $valid_scenes" | bc -l)
    mean_ssim=$(echo "$total_ssim / $valid_scenes" | bc -l)
    mean_gaussians=$((total_gaussians / valid_scenes))
    mean_time=$(echo "$total_time / $valid_scenes" | bc -l)

    mean_psnr_fmt=$(format_number $mean_psnr 4)
    mean_ssim_fmt=$(format_number $mean_ssim 6)
    mean_gaussians_fmt=$(format_with_commas $mean_gaussians)
    mean_time_fmt=$(format_duration "$mean_time")

    echo "================================================================================"
    printf "%-10s %-10s %-10s %-10s %-12s %-15s\n" \
        "mean" \
        "30000" \
        "$mean_psnr_fmt" \
        "$mean_ssim_fmt" \
        "$mean_time_fmt" \
        "$mean_gaussians_fmt"
fi

echo "================================================================================"


# Add two blank lines at the end
echo
echo
