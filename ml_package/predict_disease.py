import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

def predict_health_status(input_data):
    if not os.path.exists(os.path.join(MODEL_DIR,'disease_classifier_model.pkl')):
        return {"error": "Model files not found in models/ directory."}
    
    disease_model = joblib.load(os.path.join(MODEL_DIR,'disease_classifier_model.pkl'))
    triage_model = joblib.load(os.path.join(MODEL_DIR,'triage_risk_model.pkl'))
    escalation_model = joblib.load(os.path.join(MODEL_DIR,'escalation_model.pkl'))
    
    scaler = joblib.load(os.path.join(MODEL_DIR,'feature_scaler.pkl'))
    triage_le = joblib.load(os.path.join(MODEL_DIR,'triage_label_encoder.pkl'))
    expected_features = joblib.load(os.path.join(MODEL_DIR,'disease_features.pkl'))

    df = pd.DataFrame([input_data])
    
    cat_cols = ['Animal_Species', 'Sex', 'Recent_Treatment', 'Season']
    df_encoded = pd.get_dummies(df, columns=cat_cols)
    df_aligned = df_encoded.reindex(columns=expected_features, fill_value=0)
    
    disease_pred = disease_model.predict(df_aligned)[0]
    disease_probabilities = disease_model.predict_proba(df_aligned)[0]
    max_disease_prob = max(disease_probabilities) * 100
    ranked_diseases = sorted(
        [{"disease": str(label), "probability": float(prob)} for label, prob in zip(disease_model.classes_, disease_probabilities)],
        key=lambda x: x["probability"], reverse=True
    )
    
    triage_pred_num = triage_model.predict(df_aligned)[0]
    triage_pred_label = triage_le.inverse_transform([triage_pred_num])[0]
    
    df_scaled = scaler.transform(df_aligned)
    escalation_pred = escalation_model.predict(df_scaled)[0]
    escalation_prob = escalation_model.predict_proba(df_scaled)[0][1] * 100
    
    return {
        "Predicted_Disease": disease_pred,
        "Outbreak_Disease_Chance": f"{max_disease_prob:.1f}%",
        "Disease_Probabilities": ranked_diseases,
        "Triage_Risk_Level": triage_pred_label,
        "Outbreak_Escalation_Risk": f"{escalation_prob:.1f}%",
        "Immediate_Escalation_Required": "Yes" if escalation_pred == 1 else "No"
    }

if __name__ == "__main__":
    sample_farm_report = {
        "Animal_Species": "Cow",
        "Age_Months": 36,
        "Sex": "Female",
        "Herd_Size": 12,
        "Vaccinated_FMD": 0,
        "Vaccinated_HS": 0,
        "Vaccinated_LSD": 0,
        "Vaccinated_BQ": 0,
        "Days_Since_Last_Vaccination": 999,
        "Recent_Treatment": "Untreated",
        "Latitude": 18.5,
        "Longitude": 73.8,
        "Dist_Nearest_Outbreak_km": 5.0,
        "Dist_Waterbody_km": 1.2,
        "Temperature_C": 31.0,
        "Humidity_Percent": 85.0,
        "Rainfall_mm": 20.0,
        "Season": "Monsoon",
        "District_Outbreaks_30d": 2,
        "Symptom_Onset_Days": 2,
        "Sick_Animal_Count": 2,
        "Mortality_Count": 0,
        "Fever": 1, 
        "Salivation": 1, 
        "Blisters_Mouth_Teats": 1, 
        "Lameness": 1, 
        "Skin_Nodules": 0, 
        "Udder_Swelling": 0, 
        "Respiratory_Distress": 0, 
        "Abnormal_Milk": 0, 
        "Diarrhea": 0, 
        "Bleeding_Orifices": 0
    }
    
    results = predict_health_status(sample_farm_report)
    print("--- GRAMVET AI DIAGNOSIS ---")
    for key, value in results.items():
        print(f"{key.replace('_', ' ')}: {value}")
