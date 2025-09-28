# 🚀 Deployment Instructions

## ❌ Vercel Issue
Vercel has a 50MB limit for serverless functions, but your face recognition models are larger than this limit.

## ✅ Recommended Solution: Google Cloud Run

### Why Google Cloud Run?
- ✅ **No size limits** - Can handle large models
- ✅ **Free tier** - 2 million requests/month
- ✅ **No time limits** - Permanent free usage
- ✅ **No credit card required** for free tier
- ✅ **Auto-scaling** - Handles traffic spikes

### Step 1: Create Google Cloud Account
1. Go to [cloud.google.com](https://cloud.google.com)
2. Sign up with your Google account
3. Create a new project (or use existing)

### Step 2: Enable APIs
1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Enable "Cloud Run API"
3. Enable "Cloud Build API"

### Step 3: Deploy Using Google Cloud CLI
```bash
# Install Google Cloud CLI
# Download from: https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Deploy to Cloud Run
gcloud run deploy face-recognition-api \
    --source . \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 1 \
    --timeout 300
```

### Step 4: Get Your API URL
After deployment, you'll get a URL like:
`https://face-recognition-api-abc123-uc.a.run.app`

## 🧪 Test Your API
```bash
# Health check
curl https://your-api-url/health

# Face recognition
curl -X POST "https://your-api-url/recognize" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/test/person_1.jpg"
```

## 🎯 Alternative: Use the Lightweight Vercel Version

If you want to stick with Vercel, use the lightweight version:
- ✅ **Smaller size** - Fits Vercel limits
- ⚠️ **Demo only** - Not full recognition
- ✅ **Quick deployment** - Works immediately

The lightweight version will:
- Detect faces using OpenCV
- Return mock predictions
- Show that the API structure works
- Be ready for full deployment later

## 🚀 Next Steps

1. **Try Google Cloud Run** - Best for full functionality
2. **Or use lightweight Vercel** - Quick demo version
3. **Both are free** - No credit card required
