# 🚀 Face Recognition API Deployment Guide

## 📋 Overview
This guide will help you deploy your face recognition project as a REST API using FastAPI.

## 🏗️ Project Structure
```
face-recognition-project/
├── api.py                          # FastAPI backend
├── requirements.txt                # Python dependencies
├── Procfile                        # Railway/Heroku deployment
├── runtime.txt                     # Python version
├── .gitignore                      # Git ignore rules
├── test_api.py                     # API testing script
├── production_models/              # Trained models
│   └── face_recognizer.joblib
├── corrected_comparison_results/   # Alternative models
│   └── embeddings_mode_models/
├── data/
│   ├── embeddings/                 # Face embeddings
│   └── test/                       # Test images
└── src/                           # Source code
```

## 🧪 Step 1: Test Locally

### 1.1 Start the API
```bash
# Activate your conda environment
conda activate face-recog

# Install FastAPI if not already installed
pip install fastapi uvicorn python-multipart requests

# Start the API server
python api.py
```

The API will be available at: `http://localhost:8000`

### 1.2 Test the API
```bash
# Run the test script
python test_api.py
```

### 1.3 Manual Testing
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test face recognition
curl -X POST "http://localhost:8000/recognize" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/test/person_1.jpg"
```

## ☁️ Step 2: Deploy to Railway (Recommended)

### 2.1 Prepare for Deployment
```bash
# Make sure all files are committed
git add .
git commit -m "Add FastAPI backend for face recognition"
git push origin main
```

### 2.2 Deploy on Railway
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "New Project"
4. Select "Deploy from GitHub repo"
5. Choose your repository
6. Railway will automatically detect it's a Python app

### 2.3 Railway Configuration
Railway will automatically:
- ✅ Install dependencies from `requirements.txt`
- ✅ Use `Procfile` to start the app
- ✅ Set the `PORT` environment variable
- ✅ Provide HTTPS URL

### 2.4 Get Your API URL
After deployment, you'll get a URL like:
`https://your-app-name.railway.app`

## 🧪 Step 3: Test Deployed API

### 3.1 Test Health Endpoint
```bash
curl https://your-app-name.railway.app/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "face_detector_loaded": true,
  "num_classes": 7,
  "classes": ["ameesha", "gihan", "keshan", "lakshan", "oshanda", "pasindu", "ravishan"]
}
```

### 3.2 Test Face Recognition
```bash
curl -X POST "https://your-app-name.railway.app/recognize" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/test/person_1.jpg"
```

Expected response:
```json
{
  "success": true,
  "prediction": "gihan",
  "confidence": 0.95,
  "processing_time": 0.1,
  "faces_detected": 1,
  "top_predictions": [
    {"name": "gihan", "confidence": 0.95},
    {"name": "oshanda", "confidence": 0.03},
    {"name": "lakshan", "confidence": 0.02}
  ]
}
```

## 📊 API Endpoints

| Endpoint | Method | Purpose | Input | Output |
|----------|--------|---------|-------|--------|
| `/` | GET | Health check | None | Status message |
| `/health` | GET | Detailed health | None | Model status |
| `/classes` | GET | Get classes | None | List of recognized people |
| `/recognize` | POST | Single face recognition | Image file | Recognition result |
| `/recognize-base64` | POST | Base64 face recognition | Base64 image | Recognition result |
| `/recognize-multiple` | POST | Multiple faces recognition | Image file | Multiple results |

## 🔧 Alternative Deployment Options

### Google Cloud Run
```bash
# Create Dockerfile
# Build and deploy
gcloud run deploy face-recognition-api \
    --source . \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated
```

### Render
1. Go to [render.com](https://render.com)
2. Create new "Web Service"
3. Connect GitHub repository
4. Configure:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn api:app --host 0.0.0.0 --port $PORT`

## 🎯 What You'll Have After Deployment

### ✅ Working API with:
- **Face detection** using InsightFace
- **Face recognition** using your trained models
- **Multiple input formats** (file upload, base64)
- **Detailed responses** with confidence scores
- **Error handling** for invalid images
- **CORS enabled** for frontend access
- **Automatic documentation** at `/docs`

### ✅ Ready for Frontend:
- **REST API endpoints** for image upload
- **Base64 endpoint** for real-time frontend
- **JSON responses** easy to parse
- **HTTPS URL** for secure access

## 🚀 Next Steps

1. **Test all endpoints** with your test images
2. **Document the API** (FastAPI provides automatic docs at `/docs`)
3. **Monitor performance** and response times
4. **Prepare for Phase 2** (real-time frontend)

## 💡 Tips for Success

### Before Deployment:
- ✅ Test locally first
- ✅ Make sure all models are in the right place
- ✅ Check that `requirements.txt` has all dependencies
- ✅ Verify your `Procfile` is correct

### After Deployment:
- ✅ Test with real images
- ✅ Check response times
- ✅ Monitor for errors
- ✅ Save your API URL for Phase 2

## 🎉 Summary

**You'll have a fully functional face recognition API that:**
- ✅ **Accepts image uploads** via HTTP POST
- ✅ **Returns recognition results** with confidence scores
- ✅ **Works from anywhere** via HTTPS URL
- ✅ **Ready for frontend integration** in Phase 2
- ✅ **Completely free** to host and use

**Your API will be available at:** `https://your-app-name.railway.app`
