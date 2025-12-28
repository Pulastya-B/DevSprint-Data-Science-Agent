# Deploying to HuggingFace Spaces 🤗

This guide shows how to deploy the DevSprint Data Science Agent to HuggingFace Spaces with **16GB RAM for free** - perfect for memory-intensive data science workloads.

## Why HuggingFace Spaces?

- ✅ **16GB RAM** (vs Render's 512MB free tier)
- ✅ **Completely free** for public Spaces
- ✅ **Perfect for ML/AI demos**
- ✅ **Persistent storage** for uploaded files
- ✅ **Auto-restart** on crashes
- ✅ **Built-in secrets management**

## Prerequisites

1. **HuggingFace Account**: Sign up at https://huggingface.co/join
2. **Google Gemini API Key**: Get from https://aistudio.google.com/app/apikey
3. **Git**: Installed locally

## Quick Deployment

### Step 1: Create a New Space

1. Go to https://huggingface.co/new-space
2. Fill in details:
   - **Owner**: Your username
   - **Space name**: `devs-print-data-science-agent` (or any name)
   - **License**: MIT
   - **Select the Space SDK**: Docker
   - **Visibility**: Public (for free 16GB RAM)

3. Click **Create Space**

### Step 2: Set Up Repository

After creating the Space, HuggingFace will show you a Git repository URL like:
```
https://huggingface.co/spaces/YOUR_USERNAME/devs-print-data-science-agent
```

### Step 3: Prepare Files

**IMPORTANT**: Rename `Dockerfile.spaces` to `Dockerfile` and `README_SPACES.md` to `README.md` before pushing:

```powershell
# Backup original files
Copy-Item Dockerfile Dockerfile.render
Copy-Item README.md README_original.md

# Use HuggingFace versions
Copy-Item Dockerfile.spaces Dockerfile -Force
Copy-Item README_SPACES.md README.md -Force
```

### Step 4: Push to HuggingFace

```powershell
# Add HuggingFace remote
git remote add huggingface https://huggingface.co/spaces/YOUR_USERNAME/devs-print-data-science-agent

# Push to HuggingFace
git push huggingface main
```

**Note**: Use your HuggingFace username and access token (not password) for authentication.
- Get access token: https://huggingface.co/settings/tokens

### Step 5: Configure Secrets

1. Go to your Space: `https://huggingface.co/spaces/YOUR_USERNAME/devs-print-data-science-agent`
2. Click **Settings** tab
3. Scroll to **Repository secrets**
4. Click **New secret**
5. Add:
   - **Name**: `GEMINI_API_KEY`
   - **Value**: Your Google Gemini API key
6. Click **Save**

### Step 6: Wait for Build

HuggingFace will automatically:
1. Build your Docker container (5-10 minutes)
2. Deploy to 16GB RAM instance
3. Show build logs in the **Logs** tab
4. Start your app on port 7860

Once deployed, your Space will be live at:
```
https://YOUR_USERNAME-devs-print-data-science-agent.hf.space
```

## Dockerfile Changes for Spaces

The `Dockerfile.spaces` includes these HuggingFace-specific optimizations:

1. **Port 7860**: HuggingFace Spaces standard port
   ```dockerfile
   ENV PORT=7860
   EXPOSE 7860
   CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
   ```

2. **Non-root user**: Security requirement
   ```dockerfile
   RUN useradd -m -u 1000 user
   USER user
   WORKDIR /home/user/app
   ```

3. **User-writable directories**: For uploads and outputs
   ```dockerfile
   ENV OUTPUT_DIR=/home/user/app/outputs
   ENV CACHE_DB_PATH=/home/user/app/cache_db/cache.db
   ```

## README Metadata

The `README_SPACES.md` includes YAML frontmatter required by HuggingFace:

```yaml
---
title: DevSprint Data Science Agent
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 7860
---
```

## Troubleshooting

### Build Fails

- Check **Logs** tab for errors
- Common issue: Missing dependencies in `requirements.txt`
- Solution: Add any missing packages and push again

### App Crashes on Startup

- Check if `GEMINI_API_KEY` is set in Repository secrets
- Verify port 7860 is exposed in Dockerfile
- Check logs for Python import errors

### Memory Issues

- HuggingFace Spaces provides 16GB RAM
- Your memory optimization (sampling to 50k rows) will work great here
- For even larger datasets (>100MB), consider increasing sample size in [src/tools/eda_reports.py](src/tools/eda_reports.py)

### File Uploads Not Persisting

- Files in `/home/user/app/outputs` persist between restarts
- Temp files in `/tmp` are ephemeral (cleared on restart)
- For production, consider using HuggingFace Datasets or external storage

## Updating Your Space

To push updates:

```powershell
# Make your changes locally
git add .
git commit -m "Your update message"

# Push to both GitHub and HuggingFace
git push origin main
git push huggingface main
```

HuggingFace will automatically rebuild and redeploy.

## Comparison: Render vs HuggingFace Spaces

| Feature | Render (Free) | HuggingFace Spaces |
|---------|---------------|-------------------|
| RAM | 512MB | **16GB** ✅ |
| CPU | Shared | Shared |
| Storage | Ephemeral | Persistent |
| Cost | Free | Free (public) |
| Build Time | 3-5 min | 5-10 min |
| Auto-restart | ✅ | ✅ |
| Custom Domain | ❌ | ❌ |
| Best For | Simple APIs | **ML/Data Science** ✅ |

## Going to Production

For private Spaces with more resources:

- **HuggingFace Pro**: $9/mo for private Spaces
- **Upgraded Hardware**: Up to 32GB RAM, GPUs available
- **Custom domains**: Available with Pro

## Support

- **HuggingFace Docs**: https://huggingface.co/docs/hub/spaces-overview
- **Community Forum**: https://discuss.huggingface.co/
- **Status Page**: https://status.huggingface.co/

---

**Ready to deploy?** Follow the steps above and your agent will be live in 10-15 minutes! 🚀
