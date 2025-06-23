#!/usr/bin/env bash
# docker-launch.sh
# Adapted for EchoPulse

CONT_NAME="echopulse-app"
PORT_NUM=8501

# Cleanup any old instances
./docker-cleanup.sh

# Build the Docker image
docker build -t $CONT_NAME .

# Run the container
docker run -d \
  --name $CONT_NAME \
  -p $PORT_NUM:$PORT_NUM \
  $CONT_NAME

echo "🚀 EchoPulse is running at http://localhost:$PORT_NUM"