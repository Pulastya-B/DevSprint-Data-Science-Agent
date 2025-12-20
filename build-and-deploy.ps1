# Build and Deploy Script for Data Science Agent (Windows)

Write-Host "🚀 Building and Deploying Data Science Agent..." -ForegroundColor Cyan

# Step 1: Build React Frontend
Write-Host ""
Write-Host "📦 Building React frontend..." -ForegroundColor Yellow
Set-Location FRRONTEEEND
npm.cmd install
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Frontend npm install failed!" -ForegroundColor Red
    exit 1
}
npm.cmd run build
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Frontend build failed!" -ForegroundColor Red
    exit 1
}
Set-Location ..

Write-Host ""
Write-Host "✅ Frontend built successfully!" -ForegroundColor Green
Write-Host "   Built files are in: FRRONTEEEND\dist" -ForegroundColor Gray

# Step 2: Install Python dependencies
Write-Host ""
Write-Host "📦 Installing Python dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️ Some Python dependencies may have failed to install" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "✅ Build complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To run the application:" -ForegroundColor Cyan
Write-Host "  python src\api\app.py" -ForegroundColor White
Write-Host ""
Write-Host "Access the app at: http://localhost:8080" -ForegroundColor Green
