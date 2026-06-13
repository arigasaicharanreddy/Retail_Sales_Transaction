import joblib 
import pandas as pd
from fastapi import FastAPI 
from pydantic import BaseModel
import warnings

# Suppress warnings to keep the terminal clean
warnings.filterwarnings('ignore')

# ==========================================
# 1. Define the Data Model
# ==========================================
class Amount(BaseModel):
    Quantity: int
    Price: float
    DiscountApplied_Percent: float
    ProductCategory: str
    PaymentMethod: str

# ==========================================
# 2. Initialize FastAPI
# ==========================================
app = FastAPI(
    title="Retail Sales ML API",
    description="API for predicting total sales amount."
)

# ==========================================
# 3. Load Pre-trained Models & Columns
# ==========================================
try:
    # Loading the correct files saved from the training script
    reg_model = joblib.load('reg_model.joblib')
    model_columns = joblib.load('model_columns.joblib')
    print("✅ Models loaded successfully!")
except FileNotFoundError:
    print("❌ Warning: Joblib files not found. Ensure 'reg_model.joblib' and 'model_columns.joblib' are in the same folder.")
    reg_model = None
    model_columns = None

# ==========================================
# 4. API Endpoints
# ==========================================
@app.get("/")
def index():
    return {
        "message": "Welcome to the Retail Sales ML API!",
        "status": "Online and ready for predictions.",
        "tip": "Navigate to /docs in your browser to test the API visually."
    }
    
@app.post("/predict")
def make_predictions(amount: Amount):
    if reg_model is None or model_columns is None:
        return {"error": "Models are not loaded. Run your training script first."}

    # Step 1: Convert Pydantic object to dictionary, then to DataFrame
    input_data = pd.DataFrame([amount.model_dump()])
    
    # Step 2: One-hot encode the text columns
    input_encoded = pd.get_dummies(input_data)
    
    # Step 3: Align columns with the training data
    x_new = input_encoded.reindex(columns=model_columns, fill_value=0)
    
    # Step 4: Make the prediction
    prediction = reg_model.predict(x_new)[0]
    
    # Step 5: Return result
    return {
        "predicted_total_amount": round(float(prediction), 2),
        "currency": "USD"
    }