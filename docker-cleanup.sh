#!/usr/bin/env bash
# docker-cleanup.sh
# Adapted for EchoPulse

APP_NAME="echopulse-app"
CONTAINER_PATTERN="echopulse"

echo "🔄 Cleaning up any existing EchoPulse containers and images..."

# List current matching containers
echo "Existing containers matching '$CONTAINER_PATTERN':"
docker ps -a | grep $CONTAINER_PATTERN || true

# Stop & remove containers
echo "Stopping containers..."
docker stop $(docker ps -a -q --filter name=$CONTAINER_PATTERN) 2>/dev/null || true
echo "Removing containers..."
docker rm -f $(docker ps -a -q --filter name=$CONTAINER_PATTERN) 2>/dev/null || true

# Remove the image
echo "Removing image $APP_NAME..."
docker rmi -f $APP_NAME 2>/dev/null || true

# Prune any dangling resources
echo "Pruning unused Docker resources..."
docker system prune -f

echo "✅ Cleanup complete. Ready for a fresh build."