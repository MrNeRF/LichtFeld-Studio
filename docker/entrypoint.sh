#!/bin/bash
set -e

echo "[ENTRYPOINT] Starting LicthFeld-Studio container..."

# Define variables
PROJECT_DIR="/home/${USER}/projects/LichtFeld-Studio"
TORCH_SRC="/home/${USER}/libtorch"
TORCH_DEST="${PROJECT_DIR}/external/libtorch"


echo "[ENTRYPOINT] Copying libtorch from ${TORCH_SRC} to ${TORCH_DEST}..."
mkdir -p "${PROJECT_DIR}/external"
rm -rf "${TORCH_DEST}"
cp -r "${TORCH_SRC}" "${TORCH_DEST}"
echo "[ENTRYPOINT] libtorch copied successfully."

exec "${@:-bash}"