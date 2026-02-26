import pandas as pd
import numpy as np
import os


os.makedirs("data", exist_ok=True)

np.random.seed(42)
n = 1000

data = {
    'Musteri_ID': range(1, n + 1),
    'Aylik_Fatura': np.random.uniform(20, 120, n), 
    'Toplam_Kullanim_Ay': np.random.randint(1, 72, n), 
    'Sozlesme_Turu': np.random.choice(['Aylik', 'Yillik', 'Iki_Yillik'], n, p=[0.5, 0.3, 0.2]),
    'Internet_Turu': np.random.choice(['Fiber', 'ADSL', 'Yok'], n, p=[0.4, 0.4, 0.2]),
    'Teknik_Destek_Aramasi': np.random.choice(['Evet', 'Hayir'], n, p=[0.3, 0.7]),
    'Churn': np.random.choice([0, 1], n, p=[0.73, 0.27])  
}

df = pd.DataFrame(data)

df.loc[(df['Aylik_Fatura'] > 80) & (df['Sozlesme_Turu'] == 'Aylik'), 'Churn'] = 1

df.to_csv("data/telco_churn_data.csv", index=False)
print("Veri seti data/telco_churn_data.csv yoluna başarıyla kaydedildi!")
