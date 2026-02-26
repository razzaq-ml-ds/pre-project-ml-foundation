import joblib
import pandas as pd 
from flask import Flask, request, jsonify


app = Flask(__name__)

MODEL_PATH = "models/model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print("model and preprocessor loaded successfully!")
except FileNotFoundError as e:
    raise RuntimeError(
        "model or preprocessor not found run main.py to train and save them!"

    ) from e 


@app.route("/predict",methods=["POST"])

def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error: no data provided!"}),400
        input_df = pd.DataFrame([data])
        input_scaled = preprocessor.transform(input_df)
        prediction = int(model.predict(input_scaled)[0])
        proba = model.predict_proba(input_scaled)
        probability = [float(proba[0][0]), float(proba[0][1])]

        return jsonify({
            "prediction": int(prediction),
            "prediction_label": "Survived" if prediction == 1 else "Not Survived",
            "probability_survived": round(probability[1], 4),
            "probability_not_survived": round(probability[0], 4)    


        })
    
    except Exception as e:
        return jsonify({"error": str(e)}),500
    

@app.route("/metrics")

def metrics():
    metrics_data = {
        "accuracy": 0.85,
        "confusion_matrix": [[95, 14], [21, 49]],
        "classification_report": {
            "0" :{"precision":0.82,"recall":0.87,"f1-score":0.84},
            "1": {"precision":0.78,"recall":0.70,"f1-score":0.74}
    }

    }

    return jsonify(metrics_data)

if __name__ == "__main__":
    app.run(debug=True)




        