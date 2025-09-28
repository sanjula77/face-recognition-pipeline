# 🚀 Deploy Face Recognition API to Google Cloud Run

## Why Google Cloud Run?
- ✅ **No size limits** - Can handle your full models
- ✅ **Real file uploads** - Supports multipart form data  
- ✅ **Always works** - No weird configuration issues
- ✅ **Free tier** - $300 credit for new accounts
- ✅ **Full FastAPI** - Can use your complete API with models

## Quick Setup (5 minutes)

### Step 1: Install Google Cloud CLI
```bash
# Download from: https://cloud.google.com/sdk/docs/install
# Or use this command:
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

### Step 2: Create Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Note your Project ID

### Step 3: Deploy
```bash
# Update the project ID in deploy_cloud_run.py
# Then run:
python deploy_cloud_run.py
```

## Manual Deployment

### Step 1: Authenticate
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### Step 2: Enable APIs
```bash
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

### Step 3: Deploy
```bash
gcloud run deploy face-recognition-api \
    --source . \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8000 \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300
```

## Test Your API

Once deployed, test with:
```bash
# Health check
curl https://your-service-url.run.app/health

# Face recognition
curl -X POST "https://your-service-url.run.app/recognize" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/test/person_1.jpg"
```

## API Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health check
- `GET /classes` - List of recognized classes
- `POST /recognize` - Recognize single face
- `POST /recognize-base64` - Recognize from base64 image
- `POST /recognize-multiple` - Recognize multiple faces

## Cost
- **Free tier**: 2 million requests/month
- **After free tier**: ~$0.40 per million requests
- **Much cheaper than Vercel** for your use case

## Next Steps
1. ✅ Test your deployed API
2. ✅ Create real-time frontend
3. ✅ Set up custom domain (optional)
4. ✅ Add authentication (optional)
