#!/bin/bash
# Quick Start Script for Data Science Agent

echo "🚀 Data Science Agent - Quick Start"
echo "==================================="
echo ""

# Check if frontend is built
if [ ! -d "FRRONTEEEND/dist" ]; then
    echo "📦 Frontend not built. Building now..."
    echo ""
    
    cd FRRONTEEEND
    
    echo "Installing frontend dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install frontend dependencies!"
        exit 1
    fi
    
    echo "Building frontend..."
    npm run build
    if [ $? -ne 0 ]; then
        echo "❌ Failed to build frontend!"
        exit 1
    fi
    
    cd ..
    echo ""
    echo "✅ Frontend built successfully!"
else
    echo "✅ Frontend already built"
fi

echo ""
echo "🐍 Starting Python backend..."
echo ""
echo "Make sure you have set the following environment variables:"
echo "  - GOOGLE_API_KEY (required for Gemini)"
echo ""
echo "Starting server at http://localhost:8080"
echo "Press Ctrl+C to stop"
echo ""

python src/api/app.py
