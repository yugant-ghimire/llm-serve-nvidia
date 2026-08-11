#!/usr/bin/env bash
#
# docker-install.sh
# Installs Docker Engine + Compose plugin the way the GPU provider's
# support guide prescribes (their official get.docker.com path, with
# their dpkg retry loop for flaky package states).
#
# IMPORTANT — provider-specific rules (from their troubleshooting guide):
#   * Do NOT configure the NVIDIA container runtime (no `nvidia-ctk runtime
#     configure`, no `runtime: nvidia`, no compose GPU reservations, no
#     CUDA_VISIBLE_DEVICES). Their container runtime injects GPU access and
#     their GPU-sharing shim into every container automatically.
#   * If NVIDIA apt key issues appear, run as root:
#       apt-get install -y cuda-keyring && rm -f /etc/apt/sources.list.d/cuda.list
#
# Run with: sudo bash docker-install.sh
set -uo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Please run as root: sudo bash $0" >&2
  exit 1
fi

TARGET_USER="${SUDO_USER:-$USER}"

echo "==> Installing Docker via the provider-recommended official script..."
curl -fsSL https://get.docker.com | sh || true
for i in $(seq 1 20); do dpkg --configure -a && break; done
apt-get install -f -y

echo "==> Adding ${TARGET_USER} to the docker group (log out/in to take effect)..."
usermod -aG docker "$TARGET_USER" || true

echo "==> Versions installed:"
docker --version
docker compose version

echo
echo "Done. Log out and back in (or run 'newgrp docker') to use docker without sudo."
echo "Do NOT add GPU runtime overrides — the provider injects GPU access automatically."
