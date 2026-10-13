
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model and preprocessing information
artifact = joblib.load("hvac_model.pkl")

model = artifact["model"]
feature_columns = artifact["features"]
capping_limits = artifact["capping_limits"]


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            # Get sensor readings from the form
            data = {
                "Temperature": float(request.form["temperature"]),
                "Humidity": float(request.form["humidity"]),
                "Vibration": float(request.form["vibration"]),
                "Pressure": float(request.form["pressure"]),
                "Energy_Consumption": float(request.form["energy"]),
                "Runtime_Hours": float(request.form["runtime"])
            }

            # Validate the input
            if data["Runtime_Hours"] <= 0:
                raise ValueError("Runtime hours must be greater than zero.")

            # Apply the same IQR capping limits
            # that were saved with the model
            for col, (lower, upper) in capping_limits.items():
                data[col] = max(lower, min(data[col], upper))

            # Feature engineering: use the same formulas as training
            data["Temperature_Deviation"] = abs(
                data["Temperature"] - 27
            )

            data["Energy_Per_Runtime"] = (
                data["Energy_Consumption"] / data["Runtime_Hours"]
            )

            # Arrange columns in the exact training order
            input_df = pd.DataFrame([data])
            input_df = input_df[feature_columns]

            # Predict failure
            result = int(model.predict(input_df)[0])

            probability = float(
                model.predict_proba(input_df)[0][1]
            ) * 100

            if result == 1:
                prediction = "Potential Failure Detected"
            else:
                prediction = "Normal Operation"

        except (ValueError, KeyError) as e:
            error = str(e) or "Please enter valid sensor readings."

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
