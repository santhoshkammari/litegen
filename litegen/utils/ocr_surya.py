from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from PIL import Image
import base64
import io
import requests


from surya.ocr import run_ocr as _run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor

app = FastAPI(title="Surya OCR API", description="API for performing OCR using Surya")

# Global variables to store models
models = {
    "det_processor": None,
    "det_model": None,
    "rec_model": None,
    "rec_processor": None
}

def load_models_if_needed():
    """Lazy loading of models only when needed"""
    if models["det_processor"] is None:
        models["det_processor"] = load_det_processor()
    if models["det_model"] is None:
        models["det_model"] = load_det_model()
    if models["rec_model"] is None:
        models["rec_model"] = load_rec_model()
    if models["rec_processor"] is None:
        models["rec_processor"] = load_rec_processor()


class OCRRequest(BaseModel):
    image: str
    languages: List[str] = ["en"]




def get_surya_ocr(image_data: str, languages: List[str] = ['en']):
    """Process base64 encoded image"""
    try:
        # Load models if they haven't been loaded yet
        load_models_if_needed()

        # Handle base64 input
        if image_data.rsplit('.', 1)[-1] in ['png', 'jpg', 'jpeg']:
            image = Image.open(image_data)
        else:
            image_data = image_data.strip()
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))

        predictions = _run_ocr(
            [image],
            [languages],
            models["det_model"],
            models["det_processor"],
            models["rec_model"],
            models["rec_processor"]
        )[0]
        return predictions.model_dump()

    except base64.binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid base64 string")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ocr")
def ocr(request: OCRRequest):
    """
    Perform OCR on a base64 encoded image

    Request body:
    - image: Base64 encoded image string
    - languages: List of language codes (e.g., ["en", "hi"])
    """
    return get_surya_ocr(request.image, request.languages)


def get_surya_ocr_from_api(image_data: str, languages: List[str] = ['en'], host: str = "0.0.0.0",
                     port: int = 8000):
    url = f"http://{host}:{port}/ocr"

    if image_data.rsplit('.', 1)[-1] in ['png', 'jpg', 'jpeg']:
        with open(image_data, 'rb') as image_file:
            image_data = image_file.read()
        image = base64.b64encode(image_data).decode('utf-8')
    else:
        image = image_data

    payload = {
        "image": image,
        "languages": languages
    }
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API Error: {response.status_code} - {response.text}")


def create_surya_ocr_api(host="0.0.0.0", port=8000):
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    create_surya_ocr_api()
