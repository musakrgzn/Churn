import os
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Telekom Churn Tahmin API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "models", "churn_model_pipeline.pkl")
pipeline = joblib.load(model_path)

class MusteriVerisi(BaseModel):
    Aylik_Fatura: float
    Toplam_Kullanim_Ay: int
    Sozlesme_Turu: str
    Internet_Turu: str
    Teknik_Destek_Aramasi: str

@app.post("/tahmin_et")
def tahmin_et(veri: MusteriVerisi):
    df = pd.DataFrame([veri.model_dump()])
    tahmin = pipeline.predict(df)[0]
    ihtimal = pipeline.predict_proba(df)[0][1]
    
    sonuc = "Ayrilacak (CHURN)" if tahmin == 1 else "Kalacak"
    
    return {
        "Sistem_Tahmini": sonuc,
        "Ayrilma_Ihtimali": f"%{round(ihtimal * 100, 2)}"
    }