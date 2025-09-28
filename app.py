from fastapi import FastAPI
from mangum import Mangum

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Face Recognition API (Vercel)", "status": "healthy"}

@app.get("/health")
async def health():
    return {"status": "healthy", "api": "vercel-demo"}

@app.post("/recognize")
async def recognize():
    return {"success": True, "prediction": "demo", "confidence": 0.85, "note": "Demo version"}

# Vercel handler
handler = Mangum(app)
