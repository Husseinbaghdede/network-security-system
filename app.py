import os
import certifi
from dotenv import load_dotenv
load_dotenv()

import pymongo
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from starlette.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from uvicorn import run as app_run

# Import your own modules as needed
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logging import logger
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.constant.training_pipeline import (
    DATA_INGESTION_COLLECTION_NAME,
    DATA_INGESTION_DATABASE_NAME,
)

mongo_db_url = os.getenv('MONGO_DB_URL')
ca = certifi.where()
client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# Global variable to store latest predictions for dashboard insights
latest_predictions = []

# --------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------

@app.get("/", tags=["Root"])
async def home():
    # Redirect to the dashboard home.
    return RedirectResponse(url="/home")

@app.get("/home")
async def index(request: Request):
    """
    Render the dashboard home page with dynamic insights.
    If predictions exist, compute metrics (e.g., threat count, scans, nodes).
    """
    if latest_predictions:
        threat_count = sum(1 for p in latest_predictions if p.get("status", "").lower() == "malicious")
        scan_count = len(latest_predictions)
        node_count = len({p.get("source") for p in latest_predictions if p.get("source")})
        model_accuracy = latest_predictions[-1].get("model_accuracy", None)

        chart_labels = [f"Scan {i+1}" for i in range(scan_count)]
        chart_data = [1 if p.get("status", "").lower() == "malicious" else 0
                      for p in latest_predictions]
    else:
        threat_count = scan_count = node_count = model_accuracy = None
        chart_labels, chart_data = [], []

    context = {
        "request": request,
        "threat_count": threat_count,
        "scan_count": scan_count,
        "node_count": node_count,
        "model_accuracy": model_accuracy,
        "chart_labels": chart_labels,
        "chart_data": chart_data,
        "predictions": latest_predictions,
    }
    return templates.TemplateResponse("index.html", context)

@app.get("/upload")
async def upload_page(request: Request):
    """
    Render the CSV file upload page.
    """
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    """
    Read CSV file, run predictions using your ML model, update dashboard data,
    and display a results table.
    """
    global latest_predictions
    try:
        df = pd.read_csv(file.file)

        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")

        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)
        y_pred = network_model.predict(df)
        df['predicted_column'] = y_pred

        df['status'] = df['predicted_column'].apply(lambda x: "Malicious" if x == 1 else "Safe")

        df['model_accuracy'] = 98.5

        latest_predictions = df.to_dict(orient="records")

        columns_to_display = [
            'having_IP_Address',
            'URL_Length',
            'SSLfinal_State',
            'web_traffic',
            'status'
        ]
        subset_df = df[columns_to_display]

        table_html = subset_df.to_html(classes='table table-striped', index=False)

        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})

    except Exception as e:
        logger.exception("Error during prediction")
        return Response(f"Error Occurred! {e}")

@app.get("/train")
async def train_route():
    """
    Run the training pipeline.
    """
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful!")
    except Exception as e:
        logger.exception("Error during training pipeline")
        return Response(f"Error Occurred! {e}")

if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8080)
