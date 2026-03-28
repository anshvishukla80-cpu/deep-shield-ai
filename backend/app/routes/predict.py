from fastapi import APIRouter, UploadFile, File
from app.model.predictor import predict_file

router = APIRouter()

@router.post("/predict/")
async def predict(file: UploadFile = File(...)):
    return await predict_file(file)