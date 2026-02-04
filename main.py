from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from catboost import CatBoostRegressor

# =========================
# Load model ONCE
# =========================
model = CatBoostRegressor()
model.load_model("catboost_model.cbm")

FEATURES = [
    "Year",
    "Month",
    "Customer_Tenure_Years",
    "Industry",
    "Region",
    "Project_Type",
    "LOB",
    "Team_Size",
    "Customer_Satisfaction"
]

# =========================
# API schema
# =========================
class PredictionRequest(BaseModel):
    Year: int
    Month: int
    Customer_Tenure_Years: float | None
    Industry: str | None
    Region: str | None
    Project_Type: str | None
    LOB: str | None
    Team_Size: float | None
    Customer_Satisfaction: float | None

class PredictionResponse(BaseModel):
    predicted_profit_margin: float


# =========================
# App
# =========================
app = FastAPI(title="Customer Profitability API")


@app.post("/predict")
def predict_profit(data: PredictionRequest):

    df = pd.DataFrame(
        [[getattr(data, f) for f in FEATURES]],
        columns=FEATURES
    )

    prediction = model.predict(df)[0]

    return {
        "predicted_profit_margin": round(float(prediction), 2)
    }
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
