from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import json
import os

app = FastAPI(title="YOLO Detection API")

# CORS middleware to allow requests from Streamlit Cloud
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Streamlit Cloud URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model_path = "best.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found")
        model = YOLO(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

@app.get("/")
async def root():
    return {"message": "YOLO Detection API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate image type
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Convert to numpy array
        image_np = np.array(pil_image)
        
        # Run prediction
        results = model.predict(image_np)
        
        # Extract detection data
        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            
            detections.append({
                "class": model.names[cls_id],
                "confidence": conf,
                "bbox": xyxy
            })
        
        # Get annotated image
        pred_img = results[0].plot()  # BGR format
        pred_img_rgb = pred_img[:, :, ::-1]  # Convert to RGB
        pil_annotated = Image.fromarray(pred_img_rgb)
        
        # Convert annotated image to bytes
        img_byte_arr = io.BytesIO()
        pil_annotated.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        return {
            "detections": detections,
            "annotated_image": img_byte_arr.hex(),  # Send as hex string
            "image_format": "jpeg"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)