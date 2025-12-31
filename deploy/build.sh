
#!/bin/bash
set -e

echo "Building Docker image for Translation App..."

# Build the Docker image
docker build -t translation-app:latest .

echo "Build complete! Image: translation-app:latest"
echo ""
echo "To run locally:"
echo "  docker run -p 5000:5000 translation-app:latest"
echo ""
echo "Or use docker-compose:"
echo "  docker-compose up -d"
