
#!/bin/bash
set -e

# AWS deployment script for EC2
echo "Deploying Translation App to AWS EC2..."

# Variables (update these for your setup)
EC2_HOST="your-ec2-ip"
EC2_USER="ubuntu"
KEY_PATH="~/.ssh/your-key.pem"

echo "1. Building Docker image..."
docker build -t translation-app:latest .

echo "2. Saving Docker image..."
docker save translation-app:latest | gzip > translation-app.tar.gz

echo "3. Copying to EC2..."
scp -i $KEY_PATH translation-app.tar.gz $EC2_USER@$EC2_HOST:~/

echo "4. Deploying on EC2..."
ssh -i $KEY_PATH $EC2_USER@$EC2_HOST << 'EOF'
    # Install Docker if not present
    if ! command -v docker &> /dev/null; then
        sudo apt update
        sudo apt install -y docker.io
        sudo systemctl start docker
        sudo systemctl enable docker
        sudo usermod -aG docker $USER
    fi

    # Load and run the image
    docker load < translation-app.tar.gz
    
    # Stop existing container if running
    docker stop translation-app || true
    docker rm translation-app || true
    
    # Run new container
    docker run -d \
        --name translation-app \
        -p 80:5000 \
        --restart unless-stopped \
        -v ~/training_data:/app/training_data \
        -v ~/model_outputs:/app/model_outputs \
        translation-app:latest

    # Clean up
    rm translation-app.tar.gz

    echo "Deployment complete!"
    echo "Your app is running at http://$EC2_HOST"
EOF

# Clean up local tar file
rm translation-app.tar.gz

echo "Deployment finished!"
