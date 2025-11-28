"""
FastAPI Application for Image Classification and Preprocessing.

This module defines the API endpoints for the MLOps Lab1 Demo.
It provides functionalities for:
- Image Classification (Prediction)
- Image Resizing
- Grayscale Conversion
- Image Cropping
- Image Normalization

The application is built using FastAPI and serves a simple HTML frontend.
"""

import io
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
from logic.predictor import ImagePredictor


app = FastAPI(
    title="Image Classification API",
    description="API for image classification and preprocessing",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

predictor = ImagePredictor()


class PredictionRequest(BaseModel):
    """Request model for prediction."""

    seed: int | None = None


class PredictionResponse(BaseModel):
    """Response model for prediction."""

    predicted_class: str
    confidence: float
    all_classes: list[str]


class ResizeRequest(BaseModel):
    """Request model for image resizing."""

    width: int
    height: int


class ResizeResponse(BaseModel):
    """Response model for image resizing."""

    original_size: tuple[int, int]
    new_size: tuple[int, int]
    message: str


# Initial endpoint
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Home endpoint serving the main page.

    Returns:
        TemplateResponse: The rendered 'home.html' template.
    """
    return templates.TemplateResponse(request=request, name="home.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """
    Serve the favicon.ico file.

    Returns:
        FileResponse: The favicon image file.
    """
    return FileResponse("static/favicon.ico")


@app.get("/health")
async def health():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "service": "Image Classification API",
        "version": "1.0.0",
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(...), seed: int | None = None
):  # pylint: disable=unused-argument
    """
    Predict the class of an uploaded image.

    This endpoint accepts an image file, processes it, and returns the predicted class
    along with a confidence score. Currently, it uses a mock predictor.

    Args:
        file (UploadFile): The image file to classify.
        seed (int | None): Optional random seed for reproducibility of the mock prediction.

    Returns:
        PredictionResponse: A JSON object containing the predicted class, confidence score,
                            and a list of all possible classes.

    Raises:
        HTTPException: If an error occurs during prediction.
    """
    try:
        # For now, prediction is random (will be replaced with real model in Lab3)
        result = predictor.predict(image_path=file.filename, seed=seed)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/resize")
async def resize_image(
    file: UploadFile = File(...), width: int = 224, height: int = 224
):
    """
    Resize an uploaded image to specified dimensions.

    Args:
        file (UploadFile): The image file to resize.
        width (int): The target width in pixels (default: 224).
        height (int): The target height in pixels (default: 224).

    Returns:
        StreamingResponse: The resized image as a file download.

    Raises:
        HTTPException: If an error occurs during processing.
    """
    try:
        contents = await file.read()
        extension = file.filename.split(".")[-1].lower()
        image_format = "jpeg" if extension == "jpg" else extension
        resized_bytes = predictor.resize_image_from_bytes(
            contents, width, height, image_format
        )

        return StreamingResponse(
            io.BytesIO(resized_bytes),
            media_type=f"image/{image_format}",
            headers={
                "Content-Disposition": f"attachment; filename=resized_{file.filename}"
            },
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/grayscale")
async def convert_grayscale(file: UploadFile = File(...)):
    """
    Convert an uploaded image to grayscale.

    Args:
        file (UploadFile): The image file to convert.

    Returns:
        StreamingResponse: The grayscale image as a file download.

    Raises:
        HTTPException: If an error occurs during processing.
    """
    try:
        contents = await file.read()
        extension = file.filename.split(".")[-1].lower()
        image_format = "jpeg" if extension == "jpg" else extension
        output_bytes = predictor.convert_to_grayscale_from_bytes(contents, image_format)

        return StreamingResponse(
            io.BytesIO(output_bytes),
            media_type=f"image/{image_format}",
            headers={
                "Content-Disposition": f"attachment; filename=gray_{file.filename}"
            },
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/crop")
async def crop_image(
    file: UploadFile = File(...),
    left: int = 0,
    top: int = 0,
    right: int = 224,
    bottom: int = 224,
):
    """
    Crop an uploaded image to a specified region of interest (ROI).

    Args:
        file (UploadFile): The image file to crop.
        left (int): The left coordinate of the box (default: 0).
        top (int): The top coordinate of the box (default: 0).
        right (int): The right coordinate of the box (default: 224).
        bottom (int): The bottom coordinate of the box (default: 224).

    Returns:
        StreamingResponse: The cropped image as a file download.

    Raises:
        HTTPException: If an error occurs during processing.
    """
    try:
        contents = await file.read()
        extension = file.filename.split(".")[-1].lower()
        image_format = "jpeg" if extension == "jpg" else extension
        box = (left, top, right, bottom)
        output_bytes = predictor.crop_image_from_bytes(contents, box, image_format)

        return StreamingResponse(
            io.BytesIO(output_bytes),
            media_type=f"image/{image_format}",
            headers={
                "Content-Disposition": f"attachment; filename=cropped_{file.filename}"
            },
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/normalize")
async def get_image_stats(file: UploadFile = File(...)):
    """
    Normalize an uploaded image (contrast stretching).

    This endpoint normalizes the pixel values of the image and scales them
    to the 0-255 range for visualization.

    Args:
        file (UploadFile): The image file to analyze and normalize.

    Returns:
        StreamingResponse: The normalized image as a file download.

    Raises:
        HTTPException: If an error occurs during processing.
    """
    try:
        contents = await file.read()
        extension = file.filename.split(".")[-1].lower()
        image_format = "jpeg" if extension == "jpg" else extension
        output_bytes = predictor.normalize_image_from_bytes(contents, image_format)

        return StreamingResponse(
            io.BytesIO(output_bytes),
            media_type=f"image/{image_format}",
            headers={
                "Content-Disposition": f"attachment; filename=normalized_{file.filename}"
            },
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
