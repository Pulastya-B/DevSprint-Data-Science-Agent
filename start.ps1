# Quick Start Script for Data Science Agent

Write-Host "Data Science Agent - Quick Start" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan
Write-Host ""

# Check if frontend is built
if (-Not (Test-Path "FRRONTEEEND\dist")) {
    Write-Host "Frontend not built. Building now..." -ForegroundColor Yellow
    Write-Host ""
    
    Set-Location FRRONTEEEND
    
    Write-Host "Installing frontend dependencies..." -ForegroundColor Gray
    npm.cmd install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install frontend dependencies!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "Building frontend..." -ForegroundColor Gray
    npm.cmd run build
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to build frontend!" -ForegroundColor Red
        exit 1
    }
    
    Set-Location ..
    Write-Host ""
    Write-Host "Frontend built successfully!" -ForegroundColor Green
} else {
    Write-Host "Frontend already built" -ForegroundColor Green
}

Write-Host ""
Write-Host "Starting Python backend..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Make sure you have set the following environment variables:" -ForegroundColor Gray
Write-Host "  - GOOGLE_API_KEY (required for Gemini)" -ForegroundColor Gray
Write-Host ""
Write-Host "Starting server at http://localhost:8080" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop" -ForegroundColor Gray
Write-Host ""

py src\api\app.py
