#!/usr/bin/env pwsh

$SCENE_DIR = "data"
$RESULT_DIR = "results/benchmark"
$SCENE_LIST = @("garden", "bicycle", "stump", "bonsai", "counter", "kitchen", "room") # treehill flowers

# Check if results directory exists and prompt for deletion
if (Test-Path $RESULT_DIR) {
    Write-Host "Results directory '$RESULT_DIR' already exists."
    $reply = Read-Host "Do you want to delete it and start fresh? (y/N)"
    if ($reply -match '^[Yy]$') {
        Write-Host "Removing existing results directory..."
        Remove-Item -Path $RESULT_DIR -Recurse -Force
    } else {
        Write-Host "Keeping existing results. New results will overwrite existing ones for each scene."
    }
    Write-Host ""
}

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

    # Run training with evaluation
    & ./build/Release/LichtFeld-Studio.exe `
        -d "$SCENE_DIR/$SCENE/" `
        -o "$RESULT_DIR/$SCENE/" `
        --images "images_$DATA_FACTOR" `
        --test-every 8 `
        --eval `
        --headless `
        --save-eval-images `
        --config eval/mcmc_optimization_params.json

    Write-Host "Completed $SCENE"
    Write-Host ""
}

# Function to format numbers to specified decimal places
function Format-Number {
    param(
        [double]$num,
        [int]$decimals
    )
    return "{0:F$decimals}" -f $num
}

# Function to format numbers with thousands separators
function Format-WithCommas {
    param(
        [int]$num
    )
    return "{0:N0}" -f $num
}

# Print formatted results table
Write-Host ""
Write-Host "=============================================================================="
Write-Host "QUALITY METRICS SUMMARY"
Write-Host "=============================================================================="
Write-Host ("{0,-10} {1,-10} {2,-10} {3,-10} {4,-10} {5,-15}" -f "scene", "iteration", "psnr", "ssim", "lpips", "num_gaussians")
Write-Host "------------------------------------------------------------------------------"

# Collect and format results for each scene
$total_psnr = 0
$total_ssim = 0
$total_lpips = 0
$total_gaussians = 0
$valid_scenes = 0

foreach ($SCENE in $SCENE_LIST) {
    $csv_file = "$RESULT_DIR/$SCENE/metrics.csv"
    if (Test-Path $csv_file) {
        # Get the last line of metrics (final iteration)
        $final_metrics = Get-Content $csv_file -Tail 1
        
        # Parse CSV values
        $values = $final_metrics -split ','
        $iteration = $values[0]
        $psnr = [double]$values[1]
        $ssim = [double]$values[2]
        $lpips = [double]$values[3]
        $time_per_image = $values[4]
        $num_gaussians = [int]$values[5]
        
        # Format the numbers
        $psnr_fmt = Format-Number -num $psnr -decimals 4
        $ssim_fmt = Format-Number -num $ssim -decimals 6
        $lpips_fmt = Format-Number -num $lpips -decimals 6
        $gaussians_fmt = Format-WithCommas -num $num_gaussians
        
        # Print formatted row
        Write-Host ("{0,-10} {1,-10} {2,-10} {3,-10} {4,-10} {5,-15}" -f `
            $SCENE, `
            $iteration, `
            $psnr_fmt, `
            $ssim_fmt, `
            $lpips_fmt, `
            $gaussians_fmt)
        
        Write-Host "------------------------------------------------------------------------------"
        
        # Accumulate for mean calculation
        $total_psnr += $psnr
        $total_ssim += $ssim
        $total_lpips += $lpips
        $total_gaussians += $num_gaussians
        $valid_scenes++
    }
}

# Calculate and print mean
if ($valid_scenes -gt 0) {
    $mean_psnr = $total_psnr / $valid_scenes
    $mean_ssim = $total_ssim / $valid_scenes
    $mean_lpips = $total_lpips / $valid_scenes
    $mean_gaussians = [int]($total_gaussians / $valid_scenes)
    
    $mean_psnr_fmt = Format-Number -num $mean_psnr -decimals 4
    $mean_ssim_fmt = Format-Number -num $mean_ssim -decimals 6
    $mean_lpips_fmt = Format-Number -num $mean_lpips -decimals 6
    $mean_gaussians_fmt = Format-WithCommas -num $mean_gaussians
    
    Write-Host "=============================================================================="
    Write-Host ("{0,-10} {1,-10} {2,-10} {3,-10} {4,-10} {5,-15}" -f `
        "mean", `
        "30000", `
        $mean_psnr_fmt, `
        $mean_ssim_fmt, `
        $mean_lpips_fmt, `
        $mean_gaussians_fmt)
}

Write-Host "=============================================================================="

# Add two blank lines at the end
Write-Host ""
Write-Host ""