#!/usr/bin/env bash
#
# install-docker-gpu.sh
# Installs Docker Engine, the Compose plugin, and the NVIDIA Container Toolkit
# on Ubuntu (22.04 / 24.04). Run with: sudo bash install-docker-gpu.sh
#
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Please run as root: sudo bash $0" >&2
  exit 1
fi

# The user who invoked sudo (used for the docker group later)
TARGET_USER="${SUDO_USER:-$USER}"

echo "==> Removing old/distro Docker packages (if any)..."
apt-get remove -y docker.io docker-doc docker-compose podman-docker containerd runc 2>/dev/null || true

echo "==> Adding Docker's official APT repository..."
apt-get update
apt-get install -y ca-certificates curl
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  > /etc/apt/sources.list.d/docker.list

echo "==> Installing Docker Engine + Compose plugin..."
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "==> Adding ${TARGET_USER} to the docker group (log out/in to take effect)..."
usermod -aG docker "$TARGET_USER"

echo "==> Installing NVIDIA Container Toolkit..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y nvidia-container-toolkit

echo "==> Configuring Docker to use the NVIDIA runtime..."
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo "==> Versions installed:"
docker --version
docker compose version

echo
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> Running GPU passthrough test..."
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi || \
    echo "WARNING: GPU test failed — check that the NVIDIA driver is installed and loaded."
else
  echo "WARNING: nvidia-smi not found on the host."
  echo "Install the NVIDIA driver first (e.g. 'apt-get install nvidia-driver-550-server' then reboot),"
  echo "then re-run the GPU test:"
  echo "  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi"
fi

echo
echo "Done. Log out and back in (or run 'newgrp docker') to use docker without sudo."