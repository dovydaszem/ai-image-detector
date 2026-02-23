import io
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from .model_setup import model, preprocess, apply_gradcam

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def index():
    with open(Path(__file__).resolve().parent / "index.html", "r") as f:
        return f.read()

@app.get("/heatmap_icon.png")
def serve_heatmap_icon():
    return FileResponse("heatmap_icon.png")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        data = await file.read()
        
        if not data:
            return {"error": "uploaded file is empty"}

        img = Image.open(io.BytesIO(data))
        w, h = img.size
        img_tensor = preprocess(img)
        
        with torch.no_grad():
            logits = model(img_tensor)
            prob = torch.sigmoid(logits).item()

        heatmap = apply_gradcam(model, img_tensor, w, h)
        
        def steep_sigmoid(x, k=15):
            return 1 / (1 + np.exp(-k * (x - 0.5)))    
        
        if prob > 0.5:
            verdict = f"Likely AI generated (AI probability: {prob * 100:.0f}%)"
        else:
            verdict = f"Likely real (AI probability: {prob * 100:.0f}%)"

        return {
            "verdict": verdict,
            "is_ai": prob > 0.5,
            "heatmap": heatmap,
            "alpha": 0.5 * steep_sigmoid(prob)
        }

    except Exception as e:
        return {"error": str(e)}
