---
title: Data Science Agent
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
python_version: "3.12"
app_file: src/api/app.py
pinned: false
license: mit
---

# Data Science Agent 🤖

An intelligent **multi-agent AI system** for automated end-to-end data science workflows. Upload any dataset and watch the agent autonomously profile, clean, engineer features, train models, and generate insights—all through natural language.

## ✨ Key Features

### 🧠 Multi-Agent Architecture
- **5 Specialist Agents**: EDA, ML Modeling, Data Engineering, Visualization, Business Insights
- **Semantic Routing**: SBERT-powered agent selection based on query intent
- **Autonomous Workflows**: Full ML pipeline completion without manual intervention

### 📊 Complete ML Pipeline
- **Data Profiling**: YData profiling, statistical analysis, data quality reports
- **Data Cleaning**: Missing values, outliers, type conversion, deduplication
- **Feature Engineering**: 50+ feature types (time, interactions, aggregations, encodings)
- **Model Training**: 6 baseline models (Ridge, Lasso, Random Forest, XGBoost, LightGBM, CatBoost)
- **Hyperparameter Tuning**: Optuna-based optimization with early stopping
- **Visualizations**: Plotly dashboards, matplotlib plots, feature importance, residuals

### 🔧 Production-Ready Features
- **Real-time Progress**: SSE streaming for live workflow updates
- **Session Memory**: Maintains context across follow-up queries
- **Error Recovery**: Graceful fallbacks and parameter validation
- **Large Dataset Support**: Automatic sampling for 100K+ row datasets
- **HuggingFace Export**: Export datasets, models, and outputs directly to your HuggingFace repos

### 🔐 Authentication & Integration
- **Supabase Auth**: Secure user authentication with email/password and OAuth
- **HuggingFace Integration**: Connect your HF account to export artifacts
- **Personal Token Support**: Use your own HF write tokens for private uploads

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     React Frontend                          │
│              (Upload Dataset + Chat Interface)              │
└─────────────────────────┬───────────────────────────────────┘
                          │ SSE Stream
┌─────────────────────────▼───────────────────────────────────┐
│                    FastAPI Server                           │
│                    (Port 7860)                              │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                     Orchestrator                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Intent    │  │   Agent     │  │    Conversation     │  │
│  │  Detection  │──│  Selection  │──│      Pruning        │  │
│  │             │  │   (SBERT)   │  │  (12 exchanges)     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  5 Specialist Agents                        │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐   │
│  │    EDA    │ │ Modeling  │ │   Data    │ │   Viz     │   │
│  │   Agent   │ │  Agent    │ │Engineering│ │  Agent    │   │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘   │
│                      ┌───────────┐                          │
│                      │ Insights  │                          │
│                      │   Agent   │                          │
│                      └───────────┘                          │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    50+ Tools                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Data Profiling │ Feature Engineering │ Model Training│   │
│  │ Data Cleaning  │ Visualizations      │ NLP Analytics │   │
│  │ Time Series    │ Computer Vision     │ Business Intel│   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Usage
1. **Upload** your CSV/Excel/Parquet dataset
2. **Ask** in natural language: *"Analyze this dataset and predict the target column"*
3. **Watch** the agent autonomously execute the full ML pipeline
4. **Review** generated visualizations, model metrics, and insights

### Example Queries
```
"Profile this dataset and show data quality issues"
"Train models to predict the 'price' column"
"Generate feature importance visualizations"
"What are the key insights from this analysis?"
```

### HuggingFace Export
1. **Connect** your HuggingFace account via Settings → Add your HF token
2. **Generate** artifacts (datasets, models, visualizations)
3. **Export** directly to your HuggingFace repos from the Assets sidebar
4. **Share** your work with the ML community

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM Provider** | Mistral (mistral-large-latest) / Gemini / Groq |
| **Backend** | FastAPI + Python 3.12 |
| **Frontend** | React 19 + TypeScript + Vite + Tailwind |
| **Data Processing** | Polars (primary) + Pandas (XGBoost compatibility) |
| **ML Libraries** | Scikit-learn, XGBoost, LightGBM, CatBoost |
| **Hyperparameter Tuning** | Optuna with MedianPruner |
| **Semantic Search** | Sentence-BERT (all-MiniLM-L6-v2) |
| **Streaming** | Server-Sent Events (SSE) |
| **Authentication** | Supabase Auth |
| **Cloud Storage** | HuggingFace Hub API |

## 📁 Project Structure

```
src/
├── api/
│   └── app.py              # FastAPI endpoints + SSE streaming
├── orchestrator.py         # Main workflow orchestration (4500+ lines)
├── session_memory.py       # Context persistence across queries
├── session_store.py        # Session database management
├── storage/
│   ├── huggingface_storage.py  # HuggingFace Hub integration
│   └── artifact_store.py       # Local artifact management
├── tools/
│   ├── data_profiling.py   # YData profiling, statistics
│   ├── data_cleaning.py    # Missing values, outliers
│   ├── feature_engineering.py  # 50+ feature types
│   ├── model_training.py   # 6 baseline models + progress logging
│   ├── advanced_training.py    # Optuna hyperparameter tuning
│   ├── plotly_visualizations.py
│   ├── matplotlib_visualizations.py
│   └── tools_registry.py   # Tool definitions for LLM
├── reasoning/
│   ├── business_summary.py # Executive summaries
│   └── model_explanation.py    # Model interpretation
└── utils/
    ├── semantic_layer.py   # SBERT embeddings
    └── error_recovery.py   # Checkpoint management
```

## ⚙️ Configuration

### Environment Variables

```bash
# Required - Choose one LLM provider
MISTRAL_API_KEY=your_mistral_key      # Recommended
GEMINI_API_KEY=your_gemini_key        # Alternative
GROQ_API_KEY=your_groq_key            # Alternative

# Optional
LLM_PROVIDER=mistral                  # mistral, gemini, or groq
MAX_ITERATIONS=20                     # Max workflow steps

# Supabase (for authentication)
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key
```

### HuggingFace Spaces
Set secrets in: **Settings → Repository secrets**

## 🖥️ Local Development

```bash
# Clone repository
git clone https://github.com/your-repo/data-science-agent
cd data-science-agent

# Install Python dependencies
pip install -r requirements.txt

# Install and build frontend
cd FRRONTEEEND && npm install && npm run build && cd ..

# Set API key
export MISTRAL_API_KEY=your_key_here

# Run server
uvicorn src.api.app:app --host 0.0.0.0 --port 7860
```

## 📊 Model Training Details

### Baseline Models (Regression)
| Model | Type | Key Features |
|-------|------|--------------|
| Ridge | Linear | L2 regularization, fast |
| Lasso | Linear | L1 regularization, feature selection |
| Random Forest | Ensemble | Robust, feature importance |
| XGBoost | Gradient Boosting | High accuracy, GPU support |
| LightGBM | Gradient Boosting | Fast training, low memory |
| CatBoost | Gradient Boosting | Handles categoricals natively |

### Progress Logging
Real-time training progress with elapsed time:
```
🚀 Training 6 regression models on 140,757 samples...
[1/6] Training ridge... ✓ ridge trained in 2.3s
[2/6] Training lasso... ✓ lasso trained in 1.8s
[3/6] Training random_forest... ✓ random_forest trained in 45.2s
...
🏆 Best model: random_forest (R²=0.7585)
```

## 🔧 Recent Improvements

### Workflow Reliability
- ✅ **Autonomous Completion**: Full ML pipeline without manual confirmation
- ✅ **Smart Context Pruning**: Keeps 12 exchanges (was 4) for better memory
- ✅ **Target Column Persistence**: Injected into workflow guidance after pruning
- ✅ **Parameter Validation**: Strips invalid LLM-hallucinated parameters

### Performance
- ✅ **Real-time Progress Logging**: See model-by-model training status
- ✅ **Large Dataset Sampling**: Auto-sample to 50K rows for tuning
- ✅ **Checkpoint Clearing**: Fresh workflow for each new query

### Error Handling
- ✅ **SBERT Fallback**: Graceful keyword routing if embeddings fail
- ✅ **Tool Name Mapping**: Maps 8+ common hallucinated tool names
- ✅ **NoneType Safety**: Validates all comparison operands

### HuggingFace Integration
- ✅ **One-Click Export**: Export datasets, models, and outputs to HuggingFace
- ✅ **Personal Repos**: Auto-creates `ds-agent-data`, `ds-agent-models`, `ds-agent-outputs` repos
- ✅ **Secure Tokens**: User tokens stored securely in Supabase
- ✅ **Status Caching**: Efficient HF connection status checking

## 🐳 Docker Deployment

```dockerfile
# Multi-stage build
FROM node:20-slim AS frontend
# Build React frontend

FROM python:3.12-slim AS backend
# Install Python dependencies + copy frontend build
EXPOSE 7860
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

## 📈 Performance Benchmarks

| Dataset Size | Profiling | Training (6 models) | Total Workflow |
|--------------|-----------|---------------------|----------------|
| 10K rows | ~5s | ~30s | ~2 min |
| 50K rows | ~15s | ~2 min | ~5 min |
| 175K rows | ~45s | ~5 min | ~10 min |

## 🔮 Future Enhancements

We're actively working on exciting new features to make the Data Science Agent even more powerful:

### 🗄️ BigQuery Integration
- **Direct BigQuery Connection**: Query and analyze massive datasets directly from Google BigQuery
- **Smart Sampling**: Intelligent sampling strategies for billion-row tables
- **Cost Optimization**: Query cost estimation before execution
- **Schema Discovery**: Auto-detect tables, columns, and relationships

### 🔗 LangChain / LlamaIndex Compatibility
- **Framework Agnostic**: Use as a tool within LangChain agents or LlamaIndex pipelines
- **Custom Tool Registration**: Expose 50+ data science tools as LangChain tools
- **RAG Integration**: Combine with document retrieval for context-aware analysis
- **Memory Backends**: Support for LangChain memory stores and conversation history

### 💻 First-Class CLI Experience & Beautiful TUI
- **Rich Terminal UI**: Interactive dashboards with progress bars, tables, and charts
- **Keyboard Navigation**: Full workflow control without leaving the terminal
- **Pipeline Scripting**: Define reproducible workflows in YAML/TOML
- **Offline Mode**: Run locally without requiring a browser
- **SSH-Friendly**: Perfect for remote server analysis

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

---

**Built with ❤️ for autonomous data science**
