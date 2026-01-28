---
title: Data Science Agent
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# Data Science Agent 🤖

An intelligent AI agent for automated data science workflows, powered by Google Gemini 2.5 Flash with 82+ specialized tools for data analysis, visualization, and machine learning.

## Features

- 🔍 **Automated EDA**: YData profiling, statistical analysis, data quality reports
- 📊 **Smart Visualizations**: Plotly dashboards, matplotlib plots, interactive charts
- 🧹 **Data Cleaning**: Missing value handling, outlier detection, type conversion
- 🛠️ **Feature Engineering**: Automated feature creation, encoding, scaling
- 🤖 **ML Training**: AutoML with XGBoost, LightGBM, CatBoost, Neural Networks
- 💬 **Natural Language Interface**: Chat-based interaction for complex workflows
- 📈 **Business Intelligence**: KPI tracking, trend analysis, forecasting

## Tech Stack

- **Backend**: FastAPI + Python 3.12
- **LLM**: Google Gemini 2.5 Flash (text-based tool calling)
- **Data Processing**: Polars (high-performance dataframes)
- **Frontend**: React 19 + TypeScript + Vite
- **ML Libraries**: Scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch

## Usage

1. Upload your CSV/Excel dataset
2. Ask questions in natural language (e.g., "Generate a detailed profiling report")
3. The agent automatically selects and executes the right tools
4. View generated reports, visualizations, and insights

## Memory Optimization

For large datasets (>50k rows or >10MB), the agent automatically:
- Samples to 50,000 rows for profiling
- Enables minimal mode to reduce memory usage
- Disables expensive correlation/interaction calculations

This ensures smooth operation even with large datasets on HuggingFace's 16GB RAM.

## Environment Variables

Set `GEMINI_API_KEY` in HuggingFace Spaces settings (Settings → Repository secrets):

```
GEMINI_API_KEY=your_google_gemini_api_key_here
```

Get your API key from: https://aistudio.google.com/app/apikey

## Local Development

```bash
# Clone repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/devs-print-data-science-agent
cd devs-print-data-science-agent

# Install dependencies
pip install -r requirements.txt
npm install --prefix FRRONTEEEND

# Build frontend
cd FRRONTEEEND && npm run build && cd ..

# Set API key
export GEMINI_API_KEY=your_key_here

# Run server
uvicorn src.api.app:app --host 0.0.0.0 --port 7860
```

## Architecture

```
┌─────────────────┐
│  React Frontend │  ← User uploads data + asks questions
└────────┬────────┘
         │
┌────────▼────────┐
│  FastAPI Server │  ← Serves frontend + API endpoints
└────────┬────────┘
         │
┌────────▼────────┐
│  Orchestrator   │  ← LLM-driven tool selection & execution
└────────┬────────┘
         │
┌────────▼────────┐
│   82+ Tools     │  ← Specialized data science functions
└─────────────────┘
```

## Key Components

- **Orchestrator** ([src/orchestrator.py](src/orchestrator.py)): ReAct-based tool calling with Gemini
- **Tools Registry** ([src/tools/](src/tools/)): 82+ specialized data science tools
- **Session Memory** ([src/session_memory.py](src/session_memory.py)): Conversation history + file tracking
- **Artifact Store** ([src/storage/artifact_store.py](src/storage/artifact_store.py)): File management + metadata

## Deployment

This Space uses a **Docker** deployment for maximum compatibility:
- Base image: `python:3.12-slim`
- Multi-stage build (Node.js for frontend, Python for backend)
- Auto-exposes port 7860 for HuggingFace
- All dependencies bundled in container

## Contributing

Built for DevSprint Hackathon 2025. Contributions welcome post-hackathon!

## License

MIT License - see LICENSE file for details
