#!/bin/bash
# Quick setup script for macOS deployment prerequisites

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔧 Data Science Agent - Deployment Setup${NC}"
echo "=========================================="
echo ""

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo -e "${RED}❌ Homebrew not found${NC}"
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo -e "${GREEN}✅ Homebrew installed${NC}"
fi

# Install Docker Desktop
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}📦 Installing Docker Desktop...${NC}"
    brew install --cask docker
    echo -e "${GREEN}✅ Docker Desktop installed${NC}"
    echo -e "${YELLOW}⚠️  Please start Docker Desktop application, then run this script again${NC}"
    exit 0
else
    echo -e "${GREEN}✅ Docker installed${NC}"
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo -e "${YELLOW}⚠️  Docker is installed but not running${NC}"
    echo "Please start Docker Desktop application, then run this script again"
    exit 0
fi

# Install Google Cloud SDK
if ! command -v gcloud &> /dev/null; then
    echo -e "${YELLOW}☁️  Installing Google Cloud SDK...${NC}"
    brew install --cask google-cloud-sdk
    echo -e "${GREEN}✅ Google Cloud SDK installed${NC}"
    
    echo ""
    echo -e "${YELLOW}📝 Next steps:${NC}"
    echo "1. Restart your terminal to load gcloud"
    echo "2. Run: gcloud auth login"
    echo "3. Run: gcloud auth application-default login"
    echo "4. Run: gcloud config set project YOUR_PROJECT_ID"
    echo "5. Run: ./deploy.sh"
else
    echo -e "${GREEN}✅ Google Cloud SDK installed${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Authenticate with Google Cloud:"
echo "   ${YELLOW}gcloud auth login${NC}"
echo "   ${YELLOW}gcloud auth application-default login${NC}"
echo ""
echo "2. Set your GCP project:"
echo "   ${YELLOW}gcloud config set project YOUR_PROJECT_ID${NC}"
echo ""
echo "3. Set your API keys:"
echo "   ${YELLOW}export GROQ_API_KEY='your-groq-key'${NC}"
echo "   ${YELLOW}export GOOGLE_API_KEY='your-google-key'${NC}"
echo ""
echo "4. Deploy to Cloud Run:"
echo "   ${YELLOW}./deploy.sh${NC}"
echo ""
