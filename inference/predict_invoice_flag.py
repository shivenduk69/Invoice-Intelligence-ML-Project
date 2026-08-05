import joblib
import pandas as pd

MODEL_PATH = "models/predict_invoice_flag.pkl"
SCALER_PATH = "models/scaler.pkl"

def load_model(model_path: str = MODEL_PATH):
    """
    LOAD trained classifier model.
    """
    with open(model_path, "rb") as f:
        model=joblib.load(f)
        return model

def load_scaler(scaler_path: str = SCALER_PATH):
    """
    LOAD trained feature scaler.
    """
    with open(scaler_path, "rb") as f:
        scaler=joblib.load(f)
        return scaler

def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new vendor invoices.

    Parameters
    ----------
    input_data : dict

    Returns
    -------
    pd.DataFrame with predicted flag
    """

    model = load_model()
    scaler = load_scaler()
    input_df = pd.DataFrame(input_data)
    
    # Extract features in the correct order as trained
    features = [
        "invoice_quantity",
        "invoice_dollars",
        "Freight",
        "total_item_quantity",
        "total_item_dollars"
    ]
    
    X = input_df[features]
    X_scaled = scaler.transform(X)
    
    input_df['Predicted_Flag'] = model.predict(X_scaled).round()
    return input_df
