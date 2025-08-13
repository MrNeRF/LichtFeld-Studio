# Mip-NeRF 360 Benchmark and Timing Scripts

This directory contains scripts to benchmark and time the Mip-NeRF 360 model.

## Instructions

The following commands should be run from the root directory of this project.

1.  **Create a `data` directory:**
    ```bash
    mkdir data
    ```

2.  **Download and Unzip the dataset:**

    Download the dataset and unzip it into the `data` directory.

    ```bash
    wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip -O data/360_v2.zip
    unzip data/360_v2.zip -d data/
    ```

3.  **Run the scripts:**

    The scripts are designed to be run from the project root directory.

    ```bash
    ./scripts/benchmark_mipnerf360.sh
    ./scripts/timing_mipnerf360.sh
    ```

    ```ps
    powershell.exe -executionpolicy bypass scripts\benchmark_mipnerf360.ps1
    powershell.exe -executionpolicy bypass scripts\timing_mipnerf360.ps1
    ```