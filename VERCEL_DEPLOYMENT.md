# Vercel Deployment Guide

## ⚠️ Important Limitations

Vercel has significant limitations for this application:

### Execution Time Limits
- **Free/Hobby:** 10 seconds per request
- **Pro:** 60 seconds per request
- **Enterprise:** 300 seconds per request

### Memory Limits
- Maximum 3008 MB (Pro/Enterprise)
- May not be sufficient for large ML models

### File System
- Read-only except for `/tmp` (512 MB limit)
- Files in `/tmp` are ephemeral and cleared between invocations

### Recommendation
⚠️ **For ML/Data Science workloads, Render or Railway is recommended** over Vercel due to:
- Long-running analysis tasks (often >60s)
- Large model file sizes
- Memory requirements for ML operations
- Need for persistent storage

## If You Still Want to Try Vercel

### Prerequisites

1. A [Vercel account](https://vercel.com/) (free tier available)
2. Vercel CLI installed: `npm install -g vercel`
3. Your code pushed to GitHub

### Quick Deploy

#### Option 1: Via Vercel Dashboard (Easiest)

1. **Go to Vercel Dashboard**: https://vercel.com/dashboard

2. **Import Project:**
   - Click "Add New..." → "Project"
   - Select your GitHub repository: `Pulastya-B/DevSprint-Data-Science-Agent`

3. **Configure Build Settings:**
   - **Framework Preset:** Other
   - **Build Command:** `cd FRRONTEEEND && npm install && npm run build`
   - **Output Directory:** `FRRONTEEEND/dist`
   - **Install Command:** `pip install -r requirements.txt`

4. **Add Environment Variables:**
   ```
   GOOGLE_API_KEY=<your-api-key>
   LLM_PROVIDER=gemini
   GEMINI_MODEL=gemini-2.5-flash
   REASONING_EFFORT=medium
   CACHE_DB_PATH=/tmp/cache_db/cache.db
   OUTPUT_DIR=/tmp/outputs
   DATA_DIR=/tmp/data
   ```

5. **Deploy:**
   - Click "Deploy"
   - Wait for build to complete (~3-5 minutes)

#### Option 2: Via Vercel CLI

1. **Install Vercel CLI:**
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel:**
   ```bash
   vercel login
   ```

3. **Deploy:**
   ```bash
   cd "C:\Users\Pulastya\Videos\DS AGENTTTT"
   vercel
   ```

4. **Follow prompts:**
   - Link to existing project or create new one
   - Accept default settings
   - Add environment variables when prompted

5. **Production Deploy:**
   ```bash
   vercel --prod
   ```

### Environment Variables (Required)

Add these in Vercel Dashboard → Settings → Environment Variables:

```
GOOGLE_API_KEY=<your-gemini-api-key>
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-flash
REASONING_EFFORT=medium
CACHE_DB_PATH=/tmp/cache_db/cache.db
CACHE_TTL_SECONDS=86400
OUTPUT_DIR=/tmp/outputs
DATA_DIR=/tmp/data
MAX_PARALLEL_TOOLS=5
MAX_RETRIES=3
TIMEOUT_SECONDS=60
```

### Configuration Files

- **vercel.json** - Vercel deployment configuration
- Routes API requests to FastAPI backend
- Serves React frontend statically

### Known Issues and Workarounds

#### 1. Timeout Errors

**Issue:** Analysis tasks exceed 60-second limit

**Workarounds:**
- Use smaller datasets for testing
- Upgrade to Vercel Pro ($20/month) for 60s timeout
- Consider splitting long operations into multiple API calls
- Use background jobs (not supported on Vercel free tier)

#### 2. Memory Errors

**Issue:** ML models exceed memory limits

**Workarounds:**
- Use lighter models (e.g., LogisticRegression instead of XGBoost)
- Process smaller data chunks
- Upgrade to Vercel Pro for more memory

#### 3. Cold Starts

**Issue:** First request after idle is slow (~5-10s)

**Workarounds:**
- Use Vercel Pro for faster cold starts
- Implement warming functions (Pro/Enterprise only)

#### 4. File Storage

**Issue:** Generated reports/models are lost between requests

**Workarounds:**
- Store outputs in external storage (S3, Cloudinary)
- Use Vercel Blob Storage (paid feature)
- Accept ephemeral storage for demo purposes

### Testing Your Deployment

1. **Check deployment status:**
   ```bash
   vercel ls
   ```

2. **View logs:**
   ```bash
   vercel logs <deployment-url>
   ```

3. **Test health endpoint:**
   ```bash
   curl https://your-app.vercel.app/api/health
   ```

4. **Test with small dataset:**
   - Upload a small CSV (< 1MB, < 1000 rows)
   - Request simple analysis (avoid complex ML operations)

### Vercel vs Other Platforms

| Feature | Vercel | Render | Railway |
|---------|--------|--------|---------|
| **Best For** | Static sites, Next.js | Full-stack apps, ML | Full-stack apps |
| **Timeout (Free)** | 10s | 15min | 5min |
| **Timeout (Paid)** | 60s | ∞ | ∞ |
| **Memory (Max)** | 3008MB | 512MB-16GB | 512MB-32GB |
| **Cold Starts** | Fast | Medium | Fast |
| **Persistent Storage** | No (paid addon) | Yes | Yes |
| **Docker Support** | No | Yes | Yes |
| **Price (Hobby)** | $20/mo | $7/mo | $5/mo |

### Recommended Platform

For this Data Science Agent, we recommend:

1. **Render** (Best balance) - See [RENDER_DEPLOYMENT.md](RENDER_DEPLOYMENT.md)
   - ✅ No timeout limits
   - ✅ Docker support
   - ✅ Affordable ($7/mo starter)
   - ✅ Good for ML workloads

2. **Railway** (Alternative)
   - ✅ Good free tier
   - ✅ Persistent storage
   - ✅ Docker support
   - ⚠️ $5/mo minimum

3. **Vercel** (Not recommended for this app)
   - ❌ 60s timeout limit
   - ❌ No Docker support
   - ❌ Expensive for ML ($20/mo minimum)
   - ✅ Great for frontend-heavy apps

## Troubleshooting

### Deployment Fails

**Issue:** Build timeout during pip install

**Solution:**
- Reduce dependencies in requirements.txt
- Use lighter ML libraries
- Consider pre-building dependencies

**Issue:** "Function Payload Too Large"

**Solution:**
- Reduce package sizes
- Use `vercel.json` to exclude unnecessary files
- Consider serverless architecture redesign

### Runtime Errors

**Issue:** "Task timed out after 10.00 seconds"

**Solution:**
- Upgrade to Vercel Pro
- Optimize code for faster execution
- Use smaller datasets
- Consider using Render instead

**Issue:** "Out of memory"

**Solution:**
- Upgrade to higher memory tier
- Optimize memory usage
- Process data in chunks

## Conclusion

While Vercel deployment is possible, it's **not recommended** for this ML/Data Science application due to:

- ❌ Strict timeout limits (10s free, 60s pro)
- ❌ Memory constraints for ML models
- ❌ No persistent storage
- ❌ High cost for necessary features

**Better Alternative:** Use [Render](RENDER_DEPLOYMENT.md) for this application.

If you must use Vercel:
- Upgrade to Pro plan ($20/month minimum)
- Use only for simple datasets
- Expect frequent timeouts
- Consider it a demo/prototype only

---

**Need help with Render deployment instead?**
See [RENDER_DEPLOYMENT.md](RENDER_DEPLOYMENT.md) for a better solution.
