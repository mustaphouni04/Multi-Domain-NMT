
#!/bin/bash
set -e

echo "Deploying Translation App to AWS EC2..."
echo "This script should be run ON the EC2 instance after getting your code there."

# Step 1: Update system and install Docker (if not already installed)
echo "Step 1: Preparing EC2 instance..."

# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    sudo apt install -y docker.io
    sudo usermod -aG docker ubuntu
    newgrp docker
    echo "Docker installed successfully"
else
    echo "Docker already installed"
fi

# Verify Docker
docker --version

echo "Step 2: Building Docker image..."
# Build the Docker image (this may take several minutes for ML models)
docker build -t translation-app:latest .

echo "Step 3: Running the container..."
# Stop existing container if running
docker stop translation-app 2>/dev/null || true
docker rm translation-app 2>/dev/null || true

# Run new container
docker run -d \
    --name translation-app \
    -p 80:5000 \
    --restart unless-stopped \
    -v ~/training_data:/app/training_data \
    -v ~/model_outputs:/app/model_outputs \
    translation-app:latest

echo "Container started successfully!"
echo ""
echo "Monitoring container startup (models are downloading)..."
echo "This may take 1-5 minutes for the first run..."

# Wait a moment for container to start
sleep 10

# Show initial logs
echo "Initial container logs:"
docker logs translation-app --tail 20

echo ""
echo "Your app should be available at:"
echo "   Health check: curl http://$(curl -s ifconfig.me)/api/health"
echo "   Web interface: http://$(curl -s ifconfig.me)"
echo ""
echo "To monitor logs: docker logs -f translation-app"
echo "To stop: docker stop translation-app"
echo "To restart: docker restart translation-app"

