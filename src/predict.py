import json
import joblib
import pandas as pd
import numpy as np
import os

def load_model(model_path="models/model.pkl"):

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)

def load_scaler(scaler_path="models/scaler.pkl"):

    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    return None

def prepare_input(input_data):

    if isinstance(input_data, str):
        input_data = json.loads(input_data)

    df = pd.DataFrame([input_data])
    
 
    feature_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 
                       'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 
                       'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 
                       'V25', 'V26', 'V27', 'V28', 'Amount']
    

    df = df[feature_columns]
    
    return df

def predict(model, input_data, scaler=None, threshold=0.5):

    try:
        X = prepare_input(input_data)
        

        if scaler is not None:
            X = scaler.transform(X)
        
        y_pred = model.predict(X)[0]
        y_prob = model.predict_proba(X)[0][1] 
        

        final_pred = 1 if y_prob >= threshold else 0
        

        return {
            "prediction": "Fraud" if final_pred == 1 else "Legitimate",
            "class_id": int(final_pred),
            "probability": float(y_prob),
            "threshold": threshold,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
