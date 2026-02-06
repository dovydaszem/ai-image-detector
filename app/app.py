import io
import torch
from PIL import Image
from model_setup import model, preprocess
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, UploadFile, File

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
# When someone visits /, run the function below
def index():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        data = await file.read()
        
        if not data:
            return {"error": "uploaded file is empty"}

        # use the 'data' variable already in memory
        img = Image.open(io.BytesIO(data))
        x = preprocess(img)
        
        with torch.no_grad():
            logits = model(x)
            prob = torch.sigmoid(logits).item()
        
        if prob > 0.5:
            verdict = f"Likely AI generated (AI probability: {prob * 100:.0f}%)"
        else:
            verdict = f"Likely real (AI probability: {prob * 100:.0f}%)"

        return {
            "verdict": verdict,
            "is_ai": prob > 0.5
        }

    except Exception as e:
        return {"error": str(e)}