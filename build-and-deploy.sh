#!/bin/bash
# Build and Deploy Script for Data Science Agent

set -e  # Exit on error

echo "🚀 Building and Deploying Data Science Agent..."

# Step 1: Build React Frontend
echo ""
echo "📦 Building React frontend..."
cd FRRONTEEEND
npm.cmd install
npm.cmd run build
cd ..

# Step 2: Copy built frontend to deployment location (if needed)
echo ""
echo "✅ Frontend built successfully!"
echo "   Built files are in: FRRONTEEEND/dist"

# Step 3: Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Build complete!"
echo ""
echo "To run the application:"
echo "  1. Backend: python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8080"
echo "  2. Or use: python src/api/app.py"
echo ""
echo "Access the app at: http://localhost:8080"
