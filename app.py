import models.ml.classifier as clf
from fastapi import FastAPI
from joblib import load
from routes.v1.face_predict import app_face_predict_v1
from routes.home import app_home


app = FastAPI(title="Face ML API", description="API for face dataset ml model", version="1.0")


@app.on_event('startup')
async def load_model():
    clf.model = load('models/ml/face_dt_v1.joblib')


app.include_router(app_home)
app.include_router(app_face_predict_v1, prefix='/v1')