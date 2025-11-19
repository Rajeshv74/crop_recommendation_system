from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle
from flask import send_file
import io
from reportlab.pdfgen import canvas

app = Flask(__name__)

# ---------------- LOAD MODEL & SCALER ----------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("standscaler.pkl", "rb"))

# ---------------- DISTRICT LIST ----------------
districts = [
    "Bagalkote", "Ballari", "Belagavi", "Bengaluru Rural", "Bengaluru Urban",
    "Bidar", "Chamarajanagar", "Chikballapur", "Chikkamagaluru", "Chitradurga",
    "Dakshina Kannada", "Davanagere", "Dharwad", "Gadag", "Hassan",
    "Haveri", "Kalaburagi", "Kodagu", "Kolar", "Koppal",
    "Mandya", "Mysuru", "Raichur", "Ramanagara", "Shivamogga",
    "Tumakuru", "Udupi", "Uttara Kannada", "Vijayanagara", "Vijayapura",
    "Yadgir"
]

# ---------------- DROUGHT VULNERABILITY DATA ----------------
drought_vulnerability = {
    "Bagalkote": "High",
    "Ballari": "High",
    "Belagavi": "Moderate",
    "Bengaluru Rural": "Low",
    "Bengaluru Urban": "Low",
    "Bidar": "High",
    "Chamarajanagar": "Moderate",
    "Chikballapur": "High",
    "Chikkamagaluru": "Low",
    "Chitradurga": "High",
    "Dakshina Kannada": "Very Low",
    "Davanagere": "Moderate",
    "Dharwad": "Moderate",
    "Gadag": "High",
    "Hassan": "Moderate",
    "Haveri": "Moderate",
    "Kalaburagi": "Very High",
    "Kodagu": "Low",
    "Kolar": "High",
    "Koppal": "Very High",
    "Mandya": "Moderate",
    "Mysuru": "Moderate",
    "Raichur": "Very High",
    "Ramanagara": "Moderate",
    "Shivamogga": "Low",
    "Tumakuru": "Moderate",
    "Udupi": "Very Low",
    "Uttara Kannada": "Low",
    "Vijayanagara": "High",
    "Vijayapura": "High",
    "Yadgir": "Very High"
}

# ---------------- DROUGHT BASED CROPS ----------------
drought_based_crops = {
    "Very High": ["Pigeonpeas", "Blackgram", "Mothbeans", "Chickpea"],
    "High": ["Maize", "Cotton", "Jowar", "Groundnut"],
    "Moderate": ["Sugarcane", "Banana", "Turmeric", "Sunflower"],
    "Low": ["Paddy (Rice)", "Arecanut", "Vegetables"],
    "Very Low": ["Coconut", "Banana", "Coffee"]
}

# ---------------- CROP DICT ----------------
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
    6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
    11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate",
    15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# ---------------- HOME PAGE ----------------
@app.route("/")
def index():
    return render_template("index.html", districts=districts)

# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # JSON or FORM
        data = request.get_json() if request.is_json else request.form

        N = float(data.get("Nitrogen", 0))
        P = float(data.get("Phosphorus", 0))
        K = float(data.get("Potassium", 0))
        temp = float(data.get("Temperature", 0))
        humidity = float(data.get("Humidity", 0))
        ph = float(data.get("Ph", 0))
        rainfall = float(data.get("Rainfall", 0))
        district = data.get("District", "")
        season = data.get("Season", "")

        # ML Prediction
        features = np.array([[N, P, K, temp, humidity, ph, rainfall]])
        scaled = scaler.transform(features)
        pred_idx = model.predict(scaled)[0]
        predicted_crop = crop_dict.get(pred_idx, "Unknown")

        drought_level = drought_vulnerability.get(district, "Moderate")
        recommended = drought_based_crops.get(drought_level, [])

        result = {
            "district": district,
            "season": season,
            "predicted_crop": predicted_crop,
            "vulnerability": drought_level,
            "recommended": recommended,
            "inputs": {
                "N": N, "P": P, "K": K,
                "temp": temp, "humidity": humidity,
                "ph": ph, "rainfall": rainfall
            }
        }

        if request.is_json:
            return jsonify(result)

        return render_template("index.html", result=result, districts=districts)

    except Exception as e:
        if request.is_json:
            return jsonify({"error": str(e)}), 500
        return render_template("index.html", result={"error": str(e)}, districts=districts)
@app.route("/download_report", methods=["POST"])
def download_report():
    try:
        crop = request.form.get("Predicted", "Unknown")
        district = request.form.get("District", "")
        season = request.form.get("Season", "")

        # Create PDF in memory
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer)
        p.setFont("Helvetica", 14)

        p.drawString(100, 800, "Crop Recommendation Report")
        p.drawString(100, 770, f"District: {district}")
        p.drawString(100, 750, f"Season: {season}")
        p.drawString(100, 730, f"Predicted Crop: {crop}")
        p.drawString(100, 710, "Generated using ML + KSNDMC Data")

        p.showPage()
        p.save()

        buffer.seek(0)
        return send_file(
            buffer,
            as_attachment=True,
            download_name="crop_report.pdf",
            mimetype="application/pdf"
        )

    except Exception as e:
        return f"PDF Error: {e}", 500
    
@app.route("/compare", methods=["POST"])
def compare():
    try:
        data = request.get_json()
        cropA = data.get("a")
        cropB = data.get("b")

        if not cropA or not cropB:
            return jsonify({"error": "Missing crops"}), 400

        # Example static comparison data (you can update anytime)
        crop_info = {
            "Rice": {"Temp": "20–35°C", "Rainfall": "150–250mm", "Soil": "Clay"},
            "Maize": {"Temp": "18–27°C", "Rainfall": "50–100mm", "Soil": "Loamy"},
            "Cotton": {"Temp": "21–30°C", "Rainfall": "75–150mm", "Soil": "Black Soil"},
            "Mothbeans": {"Temp": "24–32°C", "Rainfall": "30–50mm", "Soil": "Sandy"},
            "Wheat": {"Temp": "10–20°C", "Rainfall": "50–120mm", "Soil": "Loamy"},
        }

        result = {
            "a": crop_info.get(cropA, {}),
            "b": crop_info.get(cropB, {})
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- CHART DATA ----------------
@app.route("/chart_data/<district>")
def chart_data(district):
    # Dummy example data
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    rainfall = [30,40,55,60,80,120,160,140,110,70,40,30]
    temp =      [22,24,28,30,32,34,33,32,30,28,25,23]

    return jsonify({
        "months": months,
        "rainfall": rainfall,
        "temp": temp
    })

# ---------------- DISTRICT DEFAULT WEATHER ----------------
@app.route("/district_defaults/<district>")
def district_defaults(district):
    return jsonify({
        "temp": 28,
        "humidity": 60,
        "rainfall": 500
    })


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
