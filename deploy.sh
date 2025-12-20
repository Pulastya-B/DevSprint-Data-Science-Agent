#!/bin/bash
# Manual deployment script for Google Cloud Run
# Use this for one-off deployments or CI/CD pipeline integration

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Data Science Agent - Cloud Run Deployment${NC}"
echo "=================================================="

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ Error: gcloud CLI not found. Install it from: https://cloud.google.com/sdk/install${NC}"
    exit 1
fi

# Get GCP Project ID
if [ -z "$GCP_PROJECT_ID" ]; then
    echo -e "${YELLOW}⚠️  GCP_PROJECT_ID not set. Using gcloud default project...${NC}"
    GCP_PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
    
    if [ -z "$GCP_PROJECT_ID" ]; then
        echo -e "${RED}❌ Error: No GCP project configured. Run: gcloud config set project YOUR_PROJECT_ID${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}📋 Project ID: ${GCP_PROJECT_ID}${NC}"

# Configuration
SERVICE_NAME="data-science-agent"
REGION="${CLOUD_RUN_REGION:-us-central1}"
IMAGE_NAME="gcr.io/${GCP_PROJECT_ID}/${SERVICE_NAME}"
MEMORY="${MEMORY:-4Gi}"
CPU="${CPU:-2}"
MAX_INSTANCES="${MAX_INSTANCES:-10}"
TIMEOUT="${TIMEOUT:-900}"

echo "Region: ${REGION}"
echo "Image: ${IMAGE_NAME}:latest"
echo "Memory: ${MEMORY}"
echo "CPU: ${CPU}"
echo ""

# Step 1: Enable required APIs
echo -e "${YELLOW}🔧 Step 1/5: Enabling required Google Cloud APIs...${NC}"
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com \
    secretmanager.googleapis.com \
    --project=${GCP_PROJECT_ID} \
    --quiet

echo -e "${GREEN}✅ APIs enabled${NC}"
echo ""

# Step 2: Create secrets (if not exist)
echo -e "${YELLOW}🔐 Step 2/5: Checking secrets...${NC}"

create_secret_if_not_exists() {
    local secret_name=$1
    local secret_value=$2
    
    if gcloud secrets describe ${secret_name} --project=${GCP_PROJECT_ID} &>/dev/null; then
        echo "  ℹ️  Secret ${secret_name} already exists"
    else
        if [ -n "${secret_value}" ]; then
            echo "  ➕ Creating secret: ${secret_name}"
            echo -n "${secret_value}" | gcloud secrets create ${secret_name} \
                --data-file=- \
                --project=${GCP_PROJECT_ID} \
                --quiet
        else
            echo -e "  ${YELLOW}⚠️  ${secret_name} not provided. You'll need to create it manually:${NC}"
            echo "     gcloud secrets create ${secret_name} --data-file=- --project=${GCP_PROJECT_ID}"
        fi
    fi
}

create_secret_if_not_exists "GROQ_API_KEY" "${GROQ_API_KEY}"
create_secret_if_not_exists "GOOGLE_API_KEY" "${GOOGLE_API_KEY}"

echo -e "${GREEN}✅ Secrets checked${NC}"
echo ""

# Step 3: Build container image
echo -e "${YELLOW}🏗️  Step 3/5: Building container image...${NC}"
gcloud builds submit \
    --tag ${IMAGE_NAME}:latest \
    --project=${GCP_PROJECT_ID} \
    --timeout=600s \
    .

echo -e "${GREEN}✅ Container built: ${IMAGE_NAME}:latest${NC}"
echo ""

# Step 4: Deploy to Cloud Run
echo -e "${YELLOW}🚀 Step 4/5: Deploying to Cloud Run...${NC}"

# Build the gcloud command
DEPLOY_CMD="gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME}:latest \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory ${MEMORY} \
    --cpu ${CPU} \
    --timeout ${TIMEOUT} \
    --max-instances ${MAX_INSTANCES} \
    --min-instances 0 \
    --concurrency 10 \
    --set-env-vars LLM_PROVIDER=groq,REASONING_EFFORT=medium,CACHE_TTL_SECONDS=86400,ARTIFACT_BACKEND=local \
    --project ${GCP_PROJECT_ID}"

# Add secrets if they exist
if gcloud secrets describe GROQ_API_KEY --project=${GCP_PROJECT_ID} &>/dev/null; then
    DEPLOY_CMD="${DEPLOY_CMD} --set-secrets GROQ_API_KEY=GROQ_API_KEY:latest"
fi

if gcloud secrets describe GOOGLE_API_KEY --project=${GCP_PROJECT_ID} &>/dev/null; then
    DEPLOY_CMD="${DEPLOY_CMD} --set-secrets GOOGLE_API_KEY=GOOGLE_API_KEY:latest"
fi

# Execute deployment
eval ${DEPLOY_CMD}

echo -e "${GREEN}✅ Deployment complete${NC}"
echo ""

# Step 5: Get service URL
echo -e "${YELLOW}🌐 Step 5/5: Retrieving service URL...${NC}"
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --region ${REGION} \
    --project ${GCP_PROJECT_ID} \
    --format 'value(status.url)')

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ DEPLOYMENT SUCCESSFUL!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "🌐 Service URL: ${GREEN}${SERVICE_URL}${NC}"
echo ""
echo "📝 Test endpoints:"
echo "  Health check:"
echo "    curl ${SERVICE_URL}/health"
echo ""
echo "  List tools:"
echo "    curl ${SERVICE_URL}/tools"
echo ""
echo "  Run analysis:"
echo "    curl -X POST ${SERVICE_URL}/run \\"
echo "      -F 'file=@data.csv' \\"
echo "      -F 'task_description=Analyze this dataset and predict the target column'"
echo ""
echo -e "${YELLOW}📊 View logs:${NC}"
echo "  gcloud run logs read ${SERVICE_NAME} --region ${REGION} --project ${GCP_PROJECT_ID} --limit 50"
echo ""
echo -e "${YELLOW}🔧 Manage service:${NC}"
echo "  gcloud run services describe ${SERVICE_NAME} --region ${REGION} --project ${GCP_PROJECT_ID}"
echo ""

# Save service URL to file
echo "${SERVICE_URL}" > .cloud_run_url
echo -e "${GREEN}💾 Service URL saved to .cloud_run_url${NC}"
