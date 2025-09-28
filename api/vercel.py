from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time

app = FastAPI(title="Face Recognition API (Vercel)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class_names = ['ameesha', 'gihan', 'keshan', 'lakshan', 'oshanda', 'pasindu', 'ravishan']

@app.get("/")
async def root():
    return {"message": "Face Recognition API (Vercel) is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "api_type": "vercel-lightweight", "classes": class_names}

@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_hash = hash(contents) % len(class_names)
        predicted_name = class_names[img_hash]
        confidence = 0.75 + (img_hash % 25) / 100
        
        return {
            "success": True,
            "prediction": predicted_name,
            "confidence": confidence,
            "processing_time": 0.1,
            "faces_detected": 1,
            "note": "Demo version - deploy to Google Cloud Run for full recognition"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Vercel serverless function handler
def handler(request):
    return app
