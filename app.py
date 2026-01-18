from flask import Flask, request, render_template
import numpy as np
import pickle
import os

# =========================
# Path handling (IMPORTANT)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model (1).pkl")
scaler_path = os.path.join(BASE_DIR, "standar.pkl")
minmax_path = os.path.join(BASE_DIR, "minmaxscaler.pkl")

# =========================
# Load models safely
# =========================
model = pickle.load(open(model_path, "rb"))
sc = pickle.load(open(scaler_path, "rb"))
ms = pickle.load(open(minmax_path, "rb"))

# =========================
# Flask App
# =========================
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    try:
        N = int(request.form['Nitrogen'])
        P = int(request.form['Phosporus'])
        K = int(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)

        prediction = model.predict(final_features)

        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
            6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon",
            10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
            14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
            17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
            20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }

        if prediction[0] in crop_dict:
            result = f"{crop_dict[prediction[0]]} is the best crop to be cultivated right there"
        else:
            result = "Sorry, we could not determine the best crop."

        return render_template('index.html', result=result)

    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")
if __name__ == "__main__":
    app.run(debug=True)
