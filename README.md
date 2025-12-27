# 🤖 AI-Powered Data Science Agent

> **An intelligent autonomous agent that performs end-to-end data science workflows through natural language**

Upload your dataset, describe what you want in plain English, and watch as the AI agent handles profiling, cleaning, feature engineering, model training, hyperparameter tuning, and comprehensive reporting - all automatically.

[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-4285F4?logo=google)](https://ai.google.dev/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org/)

---

## ✨ Key Features

### 🎯 **Autonomous AI Agent**
- **82+ Specialized ML Tools** organized across data profiling, cleaning, feature engineering, model training, and visualization
- **Intelligent Orchestration** with Google Gemini 2.5 Flash for function calling and decision-making
- **Session Memory** for contextual awareness across conversations
- **Smart Intent Detection** automatically classifies tasks (ML pipeline, cleaning only, visualization, etc.)
- **Error Recovery** with automatic retry logic and file tracking

### 🎨 **Modern Web Interface**
- **Beautiful React Frontend** with glassmorphism design and smooth animations
- **Interactive Chat** with file upload support (CSV, Parquet)
- **Report Viewer** to view YData profiling and Sweetviz HTML reports in-app
- **Markdown Support** for formatted responses
- **Session Management** to maintain conversation history

### 📊 **Complete ML Pipeline**
1. **Data Profiling** - Automated statistical analysis and data quality assessment
2. **Data Cleaning** - Smart missing value handling, outlier treatment, type conversion
3. **Feature Engineering** - Time-based features, encoding, interactions, statistical features
4. **Model Training** - Ridge, Lasso, Random Forest, XGBoost, LightGBM, CatBoost
5. **Hyperparameter Tuning** - Optuna-based optimization with 50+ trials
6. **Cross-Validation** - Stratified K-fold validation for robust evaluation
7. **Visualization** - Interactive Plotly dashboards and correlation heatmaps
8. **Reporting** - Comprehensive HTML reports with YData Profiling

### ⚡ **Production Ready**
- **FastAPI Backend** with async support and automatic API documentation
- **Docker Support** with multi-stage builds for optimized deployment
- **Rate Limiting** configured for Gemini API (6.5s intervals for 10 RPM limit)
- **Caching System** for faster repeated queries
- **CORS Enabled** for frontend-backend communication

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+ (for frontend)
- Google Gemini API key ([Get one here](https://ai.google.dev/))

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/Pulastya-B/DevSprint-Data-Science-Agent.git
cd DevSprint-Data-Science-Agent
```

**2. Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

**3. Install Python dependencies**
```bash
pip install -r requirements.txt
```

**4. Install frontend dependencies**
```bash
cd FRRONTEEEND
npm install
npm run build
cd ..
```

**5. Run the application**

**Windows:**
```powershell
.\start.ps1
```

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

The application will be available at **http://localhost:8080**

---

## 📖 Usage

### Web Interface

1. **Navigate to http://localhost:8080**
2. **Click "Launch Agent"** from the landing page
3. **Upload your dataset** (CSV or Parquet format)
4. **Type your request** in natural language:
   - "Generate a comprehensive report on this dataset"
   - "Train a model to predict [target_column]"
   - "Clean the data and show me visualizations"
   - "Perform feature engineering and train the best model"
5. **View results** in the chat and click "View Report" buttons to see detailed HTML reports

### Example Queries

```
📊 "Profile this dataset and tell me about data quality issues"

🧹 "Clean the missing values and handle outliers"

🎯 "Train a model to predict house prices with target column 'price'"

📈 "Generate a correlation heatmap and feature importance plot"

🔧 "Create time-based features and perform hyperparameter tuning"

📋 "Generate a comprehensive YData profiling report"
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    React Frontend (Port 8080)                │
│  Landing Page │ Chat Interface │ Report Viewer               │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Backend (Python 3.10+)                  │
│  /chat │ /run │ /outputs │ /api/health                      │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│           DataScienceCopilot Orchestrator                    │
│  • Gemini 2.5 Flash Integration                             │
│  • 82+ Specialized Tools                                     │
│  • Session Memory & Context                                  │
│  • Intelligent Intent Detection                              │
│  • Error Recovery & Loop Prevention                          │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     Tool Categories                          │
│  Profiling │ Cleaning │ Feature Engineering │ ML Training   │
│  Visualization │ EDA Reports │ Data Wrangling               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

### Frontend
- **React 19** - Modern UI library
- **TypeScript 5.8** - Type-safe development
- **Vite 6** - Lightning-fast build tool
- **Tailwind CSS** - Utility-first styling
- **Framer Motion** - Smooth animations
- **React Markdown** - Formatted responses

### Backend
- **FastAPI** - High-performance Python web framework
- **Google Gemini 2.5 Flash** - LLM for agent orchestration
- **Polars** - Fast dataframe library (10-100x faster than pandas)
- **Scikit-learn** - Classical ML algorithms
- **XGBoost / LightGBM / CatBoost** - Gradient boosting frameworks
- **Optuna** - Hyperparameter optimization
- **YData Profiling** - Automated EDA reports
- **Plotly / Matplotlib** - Interactive visualizations

### DevOps
- **Docker** - Containerization with multi-stage builds
- **Python-dotenv** - Environment variable management
- **SQLite** - Caching layer for performance

---

## 🐳 Docker Deployment

**Build and run with Docker:**

```bash
docker build -t ds-agent .
docker run -p 8080:8080 --env-file .env ds-agent
```

**Or use the deployment script:**

```bash
.\build-and-deploy.ps1  # Windows
./build-and-deploy.sh   # Linux/Mac
```

---

## 📂 Project Structure

```
.
├── FRRONTEEEND/              # React frontend
│   ├── components/           # UI components
│   │   ├── ChatInterface.tsx # Main chat interface
│   │   ├── HeroGeometric.tsx # Landing page hero
│   │   └── ...
│   ├── dist/                 # Built frontend
│   └── package.json
│
├── src/                      # Python backend
│   ├── api/
│   │   └── app.py           # FastAPI application
│   ├── orchestrator.py      # Agent orchestrator
│   ├── session_memory.py    # Session management
│   ├── tools/               # 82+ ML tools
│   │   ├── data_profiling.py
│   │   ├── data_cleaning.py
│   │   ├── feature_engineering.py
│   │   ├── model_training.py
│   │   └── ...
│   └── utils/               # Helper utilities
│
├── Dockerfile               # Multi-stage Docker build
├── requirements.txt         # Python dependencies
├── start.ps1 / start.sh    # Quick start scripts
└── README.md               # This file
```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory:

```bash
# LLM Provider Configuration
LLM_PROVIDER=gemini

# API Keys
GOOGLE_API_KEY=your_gemini_api_key_here

# Model Configuration
GEMINI_MODEL=gemini-2.5-flash

# Cache Configuration
CACHE_DB_PATH=./cache_db/cache.db
CACHE_TTL_SECONDS=86400

# Output Configuration
OUTPUT_DIR=./outputs
DATA_DIR=./data
```

---

## 🎯 Features in Detail

### Intelligent Intent Detection
The agent automatically classifies your request and applies the appropriate workflow:
- **Full ML Pipeline** - Complete end-to-end workflow with training
- **Exploratory Analysis** - Data profiling and visualization only
- **Cleaning Only** - Data quality improvements without modeling
- **Visualization Only** - Generate plots and dashboards
- **Multi-Intent** - Combine multiple tasks intelligently

### Session Memory
The agent remembers context across messages:
```
You: "Train a model on this dataset"
Agent: [Trains XGBoost model with R² = 0.85]

You: "Now try hyperparameter tuning"
Agent: [Automatically uses previous model and dataset]

You: "Cross-validate it"
Agent: [Applies CV to tuned model from context]
```

### Error Recovery
- Automatic retry with corrected parameters
- File existence validation before execution
- Recovery guidance showing last successful file
- Loop detection to prevent infinite retries

### Report Viewing
- Click "View Report" buttons to see HTML reports in-app
- Full-screen modal with professional styling
- Supports YData Profiling, Sweetviz, and custom dashboards

---

## 📊 Example Workflow

**Upload:** `earthquake_data.csv` (175K rows, 22 columns)

**Prompt:** "Train a model to predict earthquake magnitude"

**Agent Actions:**
1. ✅ Profiles dataset (175,947 rows, 22 columns)
2. ✅ Detects data quality issues (11.67% missing, outliers)
3. ✅ Drops high-missing columns (>40% missing)
4. ✅ Imputes remaining missing values with median/mode
5. ✅ Handles outliers with IQR clipping
6. ✅ Extracts time-based features (year, month, hour, cyclical)
7. ✅ Encodes categorical variables
8. ✅ Trains 6 baseline models (XGBoost wins with R² = 0.716)
9. ✅ Performs hyperparameter tuning (R² = 0.743)
10. ✅ Runs 5-fold cross-validation (RMSE = 0.167 ± 0.0005)
11. ✅ Generates YData profiling report
12. ✅ Creates interactive Plotly dashboard

**Result:** Trained and tuned XGBoost model ready for deployment!

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **Google Gemini** for powerful LLM capabilities
- **FastAPI** for excellent async Python framework
- **React** community for amazing UI libraries
- **Polars** for blazing-fast data processing
- **YData Profiling** for comprehensive EDA reports

---

## 📧 Contact

**Pulastya B**
- GitHub: [@Pulastya-B](https://github.com/Pulastya-B)
- Project: [DevSprint-Data-Science-Agent](https://github.com/Pulastya-B/DevSprint-Data-Science-Agent)

---

<div align="center">

**Built with ❤️ for DevSprint Hackathon**

⭐ Star this repo if you find it helpful!

</div>
