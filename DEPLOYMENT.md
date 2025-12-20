# 🚀 Google Cloud Run Deployment Guide

Complete guide to deploy the Data Science Agent to Google Cloud Run as a serverless API.

## 📋 Prerequisites

1. **Google Cloud Platform Account**
   - Active GCP account with billing enabled
   - Project created (or use existing project)

2. **Install Google Cloud SDK**
   ```bash
   # macOS (Homebrew)
   brew install --cask google-cloud-sdk
   
   # Or download from: https://cloud.google.com/sdk/install
   ```

3. **Authenticate with GCP**
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```

4. **Set Your Project**
   ```bash
   gcloud config set project YOUR_PROJECT_ID
   ```

---

## 🎯 Deployment Options

### Option 1: Automated Deployment (Recommended)

Use the provided deployment script for one-command deployment:

```bash
# Set required environment variables
export GCP_PROJECT_ID="your-project-id"
export GROQ_API_KEY="your-groq-api-key"
export GOOGLE_API_KEY="your-google-api-key"  # Optional for Gemini

# Run deployment script
./deploy.sh
```

**What it does:**
- ✅ Enables required GCP APIs (Cloud Build, Cloud Run, Secret Manager)
- ✅ Creates secrets for API keys
- ✅ Builds Docker container
- ✅ Deploys to Cloud Run
- ✅ Returns service URL

**Configuration options:**
```bash
# Optional: Customize deployment
export CLOUD_RUN_REGION="us-central1"  # Change region
export MEMORY="4Gi"                     # Increase memory
export CPU="2"                          # Set CPU count
export MAX_INSTANCES="10"               # Scale limit
export TIMEOUT="900"                    # Request timeout (15 min)

./deploy.sh
```

---

### Option 2: Manual Deployment

Step-by-step manual deployment for full control:

#### Step 1: Enable APIs
```bash
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  containerregistry.googleapis.com \
  secretmanager.googleapis.com
```

#### Step 2: Create Secrets
```bash
# Create GROQ API key secret
echo -n "your-groq-api-key" | gcloud secrets create GROQ_API_KEY --data-file=-

# Create Google API key secret (optional)
echo -n "your-google-api-key" | gcloud secrets create GOOGLE_API_KEY --data-file=-

# Grant Cloud Run access to secrets
PROJECT_NUMBER=$(gcloud projects describe $(gcloud config get-value project) --format="value(projectNumber)")
gcloud secrets add-iam-policy-binding GROQ_API_KEY \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

#### Step 3: Build Container
```bash
gcloud builds submit --tag gcr.io/$(gcloud config get-value project)/data-science-agent
```

#### Step 4: Deploy to Cloud Run
```bash
gcloud run deploy data-science-agent \
  --image gcr.io/$(gcloud config get-value project)/data-science-agent \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 900 \
  --max-instances 10 \
  --set-env-vars LLM_PROVIDER=groq,REASONING_EFFORT=medium \
  --set-secrets GROQ_API_KEY=GROQ_API_KEY:latest,GOOGLE_API_KEY=GOOGLE_API_KEY:latest
```

---

### Option 3: CI/CD with Cloud Build Triggers

Automated deployment on git push:

#### Step 1: Connect Repository
```bash
# Connect GitHub/GitLab/Bitbucket repository
gcloud beta builds connections create github connection-name \
  --region=us-central1
```

#### Step 2: Create Build Trigger
```bash
gcloud builds triggers create github \
  --name="deploy-data-science-agent" \
  --repo-name="Data-Science-Agent" \
  --repo-owner="Surfing-Ninja" \
  --branch-pattern="^main$" \
  --build-config="cloudbuild.yaml"
```

Now every push to `main` branch automatically deploys! 🎉

---

## 🧪 Testing the Deployment

### 1. Health Check
```bash
SERVICE_URL=$(gcloud run services describe data-science-agent \
  --region us-central1 \
  --format 'value(status.url)')

curl $SERVICE_URL/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "agent_ready": true,
  "provider": "groq",
  "tools_count": 82
}
```

### 2. List Available Tools
```bash
curl $SERVICE_URL/tools | jq
```

### 3. Profile a Dataset
```bash
curl -X POST $SERVICE_URL/profile \
  -F "file=@test_data/sample.csv"
```

### 4. Run Full Analysis
```bash
curl -X POST $SERVICE_URL/run \
  -F "file=@test_data/sample.csv" \
  -F "task_description=Analyze this dataset, detect outliers, and train a prediction model" \
  -F "target_col=target" \
  | jq
```

---

## 📊 Monitoring & Logs

### View Real-time Logs
```bash
gcloud run logs tail data-science-agent --region us-central1
```

### View Recent Logs
```bash
gcloud run logs read data-science-agent \
  --region us-central1 \
  --limit 50
```

### Cloud Console Monitoring
- Go to: https://console.cloud.google.com/run
- Click on `data-science-agent`
- View: Metrics, Logs, Revisions

---

## 💰 Cost Estimation

### Cloud Run Pricing (as of Dec 2024)
**Free Tier** (per month):
- 2 million requests
- 360,000 GB-seconds of memory
- 180,000 vCPU-seconds

**Paid Tier** (us-central1):
- CPU: $0.00002400 per vCPU-second
- Memory: $0.00000250 per GB-second
- Requests: $0.40 per million requests

**Example Cost for 4Gi Memory, 2 vCPU:**
- 1 request taking 60 seconds
  - CPU: 2 vCPU × 60s × $0.000024 = $0.00288
  - Memory: 4GB × 60s × $0.0000025 = $0.0006
  - Request: $0.0000004
  - **Total: ~$0.0035 per request**

**Monthly estimate for 1000 requests/month:**
- ~$3.50/month (well within free tier for testing!)

---

## 🔒 Security Best Practices

### 1. Enable Authentication (Production)
```bash
# Deploy with authentication required
gcloud run deploy data-science-agent \
  --no-allow-unauthenticated \
  --region us-central1 \
  --image gcr.io/PROJECT_ID/data-science-agent

# Create service account for clients
gcloud iam service-accounts create api-client

# Grant invoker role
gcloud run services add-iam-policy-binding data-science-agent \
  --member="serviceAccount:api-client@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.invoker" \
  --region us-central1
```

### 2. Use VPC Connector (For BigQuery/GCS)
```bash
# Create VPC connector
gcloud compute networks vpc-access connectors create ds-agent-connector \
  --network default \
  --region us-central1 \
  --range 10.8.0.0/28

# Deploy with VPC
gcloud run deploy data-science-agent \
  --vpc-connector ds-agent-connector \
  --region us-central1
```

### 3. Restrict API Keys
- Set **Application restrictions** in Google Cloud Console
- Whitelist only Cloud Run service URL
- Set **API restrictions** to only required APIs

---

## 🔧 Configuration Options

### Environment Variables
```bash
# Set during deployment
--set-env-vars KEY1=value1,KEY2=value2

# Available variables:
LLM_PROVIDER=groq                    # or "gemini"
REASONING_EFFORT=medium              # low, medium, high
CACHE_TTL_SECONDS=86400              # Cache lifetime
ARTIFACT_BACKEND=local               # or "gcs" for cloud storage
GCS_BUCKET_NAME=your-bucket          # If using GCS backend
OUTPUT_DIR=/tmp/outputs              # Output directory
MAX_PARALLEL_TOOLS=5                 # Concurrent tool execution
MAX_RETRIES=3                        # Tool retry attempts
TIMEOUT_SECONDS=300                  # Tool timeout
```

### Resource Limits
```bash
--memory 4Gi              # 128Mi to 32Gi
--cpu 2                   # 1 to 8 vCPU
--timeout 900             # Max 3600s (1 hour)
--max-instances 10        # Scale limit
--min-instances 0         # Always-warm instances
--concurrency 10          # Requests per instance
```

---

## 🐛 Troubleshooting

### Build Fails
```bash
# Check build logs
gcloud builds list --limit=5
gcloud builds log BUILD_ID

# Common fixes:
# - Ensure Dockerfile is in root directory
# - Check requirements.txt has all dependencies
# - Increase build timeout: --timeout=1200s
```

### Deployment Fails
```bash
# Check service status
gcloud run services describe data-science-agent --region us-central1

# Common fixes:
# - Ensure APIs are enabled
# - Check secrets exist and are accessible
# - Verify service account permissions
```

### Runtime Errors
```bash
# View logs
gcloud run logs tail data-science-agent --region us-central1

# Common issues:
# - API keys not set: Check secrets
# - Import errors: Ensure all dependencies in requirements.txt
# - Memory issues: Increase --memory limit
# - Timeout: Increase --timeout value
```

### Container Crashes
```bash
# Test locally first
docker build -t ds-agent .
docker run -p 8080:8080 \
  -e GROQ_API_KEY="your-key" \
  ds-agent

curl http://localhost:8080/health
```

---

## 🚀 Advanced Features

### Custom Domain
```bash
# Map custom domain
gcloud run domain-mappings create \
  --service data-science-agent \
  --domain api.yourdomain.com \
  --region us-central1
```

### Load Balancing
```bash
# Create multiple regional deployments
for region in us-central1 us-east1 europe-west1; do
  gcloud run deploy data-science-agent \
    --image gcr.io/PROJECT_ID/data-science-agent \
    --region $region
done

# Set up global load balancer
# Follow: https://cloud.google.com/load-balancing/docs/https/setup-global-ext-https-serverless
```

### Multi-Region Deployment
```bash
# Deploy to multiple regions for high availability
./deploy.sh CLOUD_RUN_REGION=us-central1
./deploy.sh CLOUD_RUN_REGION=europe-west1
./deploy.sh CLOUD_RUN_REGION=asia-east1
```

---

## 📝 API Documentation

Once deployed, access Swagger docs at:
```
https://YOUR_SERVICE_URL/docs
```

### Available Endpoints

#### `GET /` - Health Check
Returns service status and tool count.

#### `GET /health` - Detailed Health
Returns agent readiness and provider info.

#### `GET /tools` - List Tools
Returns all 82 available tools organized by category.

#### `POST /run` - Run Full Analysis
Upload dataset and execute complete data science workflow.

**Parameters:**
- `file`: CSV/Parquet file (multipart/form-data)
- `task_description`: Natural language task description
- `target_col`: Target column for ML (optional)
- `use_cache`: Enable caching (default: true)
- `max_iterations`: Max workflow steps (default: 20)

#### `POST /profile` - Quick Profile
Quick dataset profiling without full workflow.

**Parameters:**
- `file`: CSV/Parquet file (multipart/form-data)

---

## 🔄 Updates & Rollbacks

### Update Deployment
```bash
# Rebuild and redeploy
./deploy.sh
```

### Rollback to Previous Revision
```bash
# List revisions
gcloud run revisions list --service data-science-agent --region us-central1

# Rollback
gcloud run services update-traffic data-science-agent \
  --to-revisions REVISION_NAME=100 \
  --region us-central1
```

### Blue/Green Deployment
```bash
# Deploy new version with tag
gcloud run deploy data-science-agent \
  --tag blue \
  --no-traffic \
  --region us-central1

# Test: https://blue---data-science-agent-HASH.run.app

# Switch traffic
gcloud run services update-traffic data-science-agent \
  --to-tags blue=100 \
  --region us-central1
```

---

## 📚 Additional Resources

- **Cloud Run Docs**: https://cloud.google.com/run/docs
- **Pricing Calculator**: https://cloud.google.com/products/calculator
- **Best Practices**: https://cloud.google.com/run/docs/tips
- **Quotas & Limits**: https://cloud.google.com/run/quotas

---

## ✅ Deployment Checklist

- [ ] GCP project created and billing enabled
- [ ] Google Cloud SDK installed and authenticated
- [ ] API keys obtained (GROQ_API_KEY, GOOGLE_API_KEY)
- [ ] Secrets created in Secret Manager
- [ ] Docker container builds successfully locally
- [ ] Cloud Run APIs enabled
- [ ] Service deployed to Cloud Run
- [ ] Health check endpoint returns 200
- [ ] Test dataset profiled successfully
- [ ] Full analysis workflow tested
- [ ] Monitoring/logging configured
- [ ] Cost alerts set up (optional)
- [ ] Custom domain mapped (optional)
- [ ] CI/CD pipeline configured (optional)

---

**Need help?** Check the troubleshooting section or view logs with:
```bash
gcloud run logs tail data-science-agent --region us-central1
```

Happy deploying! 🎉
