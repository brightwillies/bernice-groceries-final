from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import os
import time

app = FastAPI(title="YOLO Detection API")

# CORS middleware to allow requests from Streamlit Cloud
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
model_loading = False

def load_yolo_model():
    global model, model_loading
    try:
        model_loading = True
        model_path = "best.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found")
        
        # Load model
        model = YOLO(model_path)
        print("✅ YOLO model loaded successfully")
        model_loading = False
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model_loading = False
        return False

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    print("🚀 Starting up YOLO Detection API...")
    load_yolo_model()

@app.get("/")
async def root():
    return {
        "message": "YOLO Detection API is running",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)"
        }
    }

@app.get("/health")
async def health_check():
    if model is None and not model_loading:
        # Try to load model if it's not loaded
        load_yolo_model()
    
    return {
        "status": "healthy" if model is not None else "loading",
        "model_loaded": model is not None,
        "timestamp": time.time()
    }

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # Check if model is loaded
    if model is None:
        if model_loading:
            raise HTTPException(status_code=503, detail="Model is still loading, please try again in a moment")
        else:
            # Try to load model
            if not load_yolo_model():
                raise HTTPException(status_code=503, detail="Model failed to load")
    
    # Validate image type
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and validate image
        image_data = await image.read()
        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
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
        pil_annotated.save(img_byte_arr, format='JPEG', quality=85)
        img_byte_arr = img_byte_arr.getvalue()
        
        return {
            "detections": detections,
            "annotated_image": img_byte_arr.hex(),
            "image_format": "jpeg",
            "detection_count": len(detections)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)