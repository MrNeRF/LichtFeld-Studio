#!/usr/bin/env pwsh

$SCENE_DIR = "data"
$RESULT_DIR = "results/benchmark"
$SCENE_LIST = @("garden", "bicycle", "stump", "bonsai", "counter", "kitchen", "room") # treehill flowers

# Start total timer
$total_start = Get-Date

# Initialize hashtable to store individual times
$scene_times = @{}

foreach ($SCENE in $SCENE_LIST) {
    # Determine data factor based on scene type
    if ($SCENE -in @("bonsai", "counter", "kitchen", "room")) {
        $DATA_FACTOR = 2
    } else {
        $DATA_FACTOR = 4
    }
    
    Write-Host "========================================="
    Write-Host "Running $SCENE with images_$DATA_FACTOR"
    Write-Host "========================================="
    
    # Start timer for this scene
    $scene_start = Get-Date
    
    # Run training with evaluation
    & ./build/Release/LichtFeld-Studio.exe `
        -d "$SCENE_DIR/$SCENE/" `
        -o "$RESULT_DIR/$SCENE/" `
        --images "images_$DATA_FACTOR" `
        --iter 30000 `
        --headless
		--config eval/mcmc_optimization_params.json

    # End timer for this scene
    $scene_end = Get-Date
    $scene_duration = $scene_end - $scene_start
    $scene_times[$SCENE] = $scene_duration
    
    # Format time nicely
    $hours = [int]$scene_duration.TotalHours
    $minutes = $scene_duration.Minutes
    $seconds = $scene_duration.Seconds
    
    Write-Host "Completed $SCENE in ${hours}h ${minutes}m ${seconds}s"
    Write-Host ""
}

# End total timer
$total_end = Get-Date
$total_duration = $total_end - $total_start

# Print summary
Write-Host "========================================="
Write-Host "BENCHMARK SUMMARY"
Write-Host "========================================="

# Print individual scene times
foreach ($SCENE in $SCENE_LIST) {
    $duration = $scene_times[$SCENE]
    $hours = [int]$duration.TotalHours
    $minutes = $duration.Minutes
    $seconds = $duration.Seconds
    Write-Host ("{0,-15}: {1:D2}h {2:D2}m {3:D2}s" -f $SCENE, $hours, $minutes, $seconds)
}

Write-Host "-----------------------------------------"

# Print total time
$total_hours = [int]$total_duration.TotalHours
$total_minutes = $total_duration.Minutes
$total_seconds = $total_duration.Seconds
Write-Host "Total time: ${total_hours}h ${total_minutes}m ${total_seconds}s"
Write-Host "========================================="