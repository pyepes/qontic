#!/bin/bash
set -e

REMOTE_USER="pyepes"
REMOTE_HOST="bonner-gpu.rice.edu"
REMOTE_PATH="/var/www/html/bonner-gpu/bm/2slit-wavepacket"

echo "Creating remote directory if needed..."
ssh "$REMOTE_USER@$REMOTE_HOST" "mkdir -p $REMOTE_PATH"

echo "Deploying to $REMOTE_HOST:$REMOTE_PATH ..."
rsync -avz --delete \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='.vscode' \
  --exclude='figures_run.log' \
  --exclude='*.m' \
  --exclude='*.m:Zone.Identifier' \
  --exclude='*.py' \
  --exclude='*.json' \
  --exclude='deploy.sh' \
  ./ "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/"

echo "Done. Live at: https://qonticlab.rice.edu/2slit-wavepacket/"
