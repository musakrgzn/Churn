import os
import io
import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Telekom Churn Tahmin API (Dashboard Sürümü)")

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
    risk_faktorleri = ["Sistem analizini tamamladı. Detaylar için Dashboard'u kullanın."]
    
    return {
        "Sistem_Tahmini": sonuc,
        "Ayrilma_Ihtimali": f"%{round(ihtimal * 100, 2)}",
        "Risk_Faktorleri": risk_faktorleri
    }


@app.post("/toplu_tahmin")
async def toplu_tahmin(dosya: UploadFile = File(...)):
    
    icerik = await dosya.read()
    df = pd.read_csv(io.BytesIO(icerik))
    
    
    tahminler = pipeline.predict(df)
    ihtimaller = pipeline.predict_proba(df)[:, 1]
    
    
    df['Tahmin_Sonucu'] = ["Ayrilacak" if t == 1 else "Kalacak" for t in tahminler]
    df['Risk_Yuzdesi'] = (ihtimaller * 100).round(1)
    
    
    return df.to_dict(orient="records")
