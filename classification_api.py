# # classification_api.py
# from fastapi import FastAPI
# from fastapi import HTTPException
# from pydantic import BaseModel
# import joblib
# import pandas as pd
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Or restrict to your backend URL later
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load saved model and preprocessor
# clf = joblib.load("transaction_classifier.pkl")
# preprocessor = joblib.load("preprocessor.pkl")

# class Transaction(BaseModel):
#     reference: str
#     remarks: str
#     debit: float
#     credit: float

# @app.post("/predict")
# def predict(tx: Transaction):
#     try:
#         text = (tx.reference or '') + ' ' + (tx.remarks or '')
#         input_df = pd.DataFrame([{'text': text, 'debit': tx.debit, 'credit': tx.credit}])
#         features = preprocessor.transform(input_df)
#         pred = clf.predict(features)[0]
#         return {"predicted_category": pred}
#     except Exception as e:
#         # Log or print the error
#         print("Error during prediction:", e)
#         raise HTTPException(status_code=500, detail=str(e))


from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os

API_TOKEN = os.getenv("API_TOKEN")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model & preprocessor
clf = joblib.load("transaction_classifier.pkl")
preprocessor = joblib.load("preprocessor.pkl")

class Transaction(BaseModel):
    reference: str
    remarks: str
    debit: float
    credit: float

@app.post("/predict")
def predict(
    tx: Transaction,
    authorization: str = Header(None)
):
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=403, detail="Invalid or missing token")

    try:
        text = f"{tx.reference} {tx.remarks}".strip()
        df = pd.DataFrame([{
            "text": text,
            "debit": tx.debit,
            "credit": tx.credit
        }])

        features = preprocessor.transform(df)
        pred = clf.predict(features)[0]

        return {"predicted_category": pred}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
