# Data Science Agent - Frontend Integration Guide

## 🎉 New React Frontend

The application now features a modern, professional React frontend that replaces the old Gradio interface.

### Features

- **Beautiful Landing Page**: Showcases the agent's capabilities with modern design
- **Professional Chat Interface**: NextChat-style conversational UI
- **Direct Backend Integration**: Communicates with your FastAPI backend
- **Responsive Design**: Works on all devices
- **Dark Theme**: Modern, eye-friendly interface

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- Node.js 20+
- npm (comes with Node.js)

### Running the Application

#### Option 1: Using the Build Script (Recommended)

**Windows:**
```powershell
.\build-and-deploy.ps1
```

**Linux/Mac:**
```bash
chmod +x build-and-deploy.sh
./build-and-deploy.sh
```

Then start the server:
```bash
python src/api/app.py
```

#### Option 2: Manual Steps

1. **Build the Frontend:**
```bash
cd FRRONTEEEND
npm.cmd install
npm.cmd run build
cd ..
```

2. **Install Python Dependencies:**
```bash
pip install -r requirements.txt
```

3. **Start the Backend Server:**
```bash
python src/api/app.py
```

4. **Access the Application:**
Open your browser and navigate to: http://localhost:8080

## 🏗️ Architecture

### Backend (FastAPI)
- **Location**: `src/api/app.py`
- **Port**: 8080
- **Endpoints**:
  - `GET /` - Health check & landing page
  - `POST /chat` - Chat interface endpoint
  - `POST /run` - Full data science workflow
  - `POST /profile` - Dataset profiling
  - `GET /tools` - List available tools

### Frontend (React + Vite)
- **Location**: `FRRONTEEEND/`
- **Build Output**: `FRRONTEEEND/dist/`
- **Dev Port**: 3000 (development mode)
- **Production**: Served by FastAPI at port 8080

## 🔧 Development Mode

If you want to develop the frontend with hot-reloading:

1. **Terminal 1 - Backend:**
```bash
python src/api/app.py
```

2. **Terminal 2 - Frontend:**
```bash
cd FRRONTEEEND
npm.cmd run dev
```

Access:
- Frontend (dev): http://localhost:3000
- Backend API: http://localhost:8080

## 🌐 API Integration

The frontend now communicates with your FastAPI backend instead of calling external APIs directly.

### Environment Variables

Create `FRRONTEEEND/.env` for local development:
```env
VITE_API_URL=http://localhost:8080
```

For production, update `FRRONTEEEND/.env.production`:
```env
VITE_API_URL=https://your-cloud-run-url.run.app
```

## 📦 Deployment

### Docker Build

The Dockerfile now includes a multi-stage build that:
1. Builds the React frontend
2. Builds the Python environment
3. Combines both in the final image

```bash
docker build -t data-science-agent .
docker run -p 8080:8080 data-science-agent
```

### Google Cloud Run

```bash
gcloud builds submit --tag gcr.io/YOUR-PROJECT-ID/data-science-agent
gcloud run deploy data-science-agent \
  --image gcr.io/YOUR-PROJECT-ID/data-science-agent \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GROQ_API_KEY=your-api-key
```

## 🔄 What Changed

### Removed
- ❌ Gradio interface (`chat_ui.py` - kept for reference)
- ❌ Direct Google GenAI calls from frontend
- ❌ Gradio dependency

### Added
- ✅ React + TypeScript frontend with Vite
- ✅ Professional landing page
- ✅ Modern chat interface
- ✅ `/chat` API endpoint
- ✅ CORS support in FastAPI
- ✅ Static file serving for React app
- ✅ Multi-stage Docker build

## 🛠️ Tech Stack

### Frontend
- React 19
- TypeScript 5.8
- Vite 6
- Tailwind CSS
- Framer Motion (animations)
- Lucide React (icons)

### Backend (unchanged)
- FastAPI
- Python 3.13
- Groq API
- Polars, DuckDB
- Scikit-learn, XGBoost, LightGBM

## 📁 Project Structure

```
.
├── FRRONTEEEND/              # React frontend
│   ├── components/           # React components
│   ├── dist/                 # Built frontend (after npm run build)
│   ├── package.json
│   ├── vite.config.ts
│   └── .env                  # Frontend environment variables
├── src/
│   ├── api/
│   │   └── app.py           # FastAPI backend (updated)
│   ├── tools/               # Data science tools
│   └── orchestrator.py      # Main agent logic
├── requirements.txt          # Python dependencies (updated)
├── Dockerfile               # Multi-stage build (updated)
├── build-and-deploy.ps1     # Windows build script
└── build-and-deploy.sh      # Linux/Mac build script
```

## 🐛 Troubleshooting

### Frontend doesn't load
- Make sure you've run `npm run build` in the FRRONTEEEND directory
- Check that `FRRONTEEEND/dist/` exists and contains files

### API errors in chat
- Ensure the backend is running on port 8080
- Check that `GROQ_API_KEY` is set in your environment
- Verify the API URL in `.env` file

### CORS errors
- The backend now has CORS enabled for development
- For production, update the `allow_origins` in `src/api/app.py`

## 📝 Notes

- The old `chat_ui.py` has been kept for reference but is no longer used
- All chat functionality now goes through the `/chat` endpoint
- The frontend is automatically served by FastAPI in production mode
- Session history is maintained in the frontend (browser)

## 🎯 Next Steps

1. **Customize the frontend**: Edit files in `FRRONTEEEND/components/`
2. **Add file upload**: Extend `ChatInterface.tsx` to handle file uploads
3. **Add visualization**: Display charts from the backend in the chat
4. **Authentication**: Add user authentication if needed

## 📞 Support

For issues or questions:
1. Check the console logs (browser & terminal)
2. Verify environment variables
3. Ensure all dependencies are installed
4. Review the API documentation at http://localhost:8080/docs
