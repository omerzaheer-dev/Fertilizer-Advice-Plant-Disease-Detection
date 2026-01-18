# from flask import Flask, request, render_template
# import numpy as np
# import pickle
# import os

# # =========================
# # Path handling (IMPORTANT)
# # =========================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# model_path = os.path.join(BASE_DIR, "model (1).pkl")
# scaler_path = os.path.join(BASE_DIR, "standar.pkl")
# minmax_path = os.path.join(BASE_DIR, "minmaxscaler.pkl")

# # =========================
# # Load models safely
# # =========================
# model = pickle.load(open(model_path, "rb"))
# sc = pickle.load(open(scaler_path, "rb"))
# ms = pickle.load(open(minmax_path, "rb"))

# # =========================
# # Flask App
# # =========================
# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route("/predict", methods=['POST'])
# def predict():
#     try:
#         N = int(request.form['Nitrogen'])
#         P = int(request.form['Phosporus'])
#         K = int(request.form['Potassium'])
#         temp = float(request.form['Temperature'])
#         humidity = float(request.form['Humidity'])
#         ph = float(request.form['Ph'])
#         rainfall = float(request.form['Rainfall'])

#         feature_list = [N, P, K, temp, humidity, ph, rainfall]
#         single_pred = np.array(feature_list).reshape(1, -1)

#         scaled_features = ms.transform(single_pred)
#         final_features = sc.transform(scaled_features)

#         prediction = model.predict(final_features)

#         crop_dict = {
#             1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
#             6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon",
#             10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
#             14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
#             17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
#             20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
#         }

#         if prediction[0] in crop_dict:
#             result = f"{crop_dict[prediction[0]]} is the best crop to be cultivated right there"
#         else:
#             result = "Sorry, we could not determine the best crop."

#         return render_template('index.html', result=result)

#     except Exception as e:
#         return render_template('index.html', result=f"Error: {str(e)}")
# if __name__ == "__main__":
#     app.run(debug=True)


















from flask import Flask, request, render_template
import numpy as np
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------
# Path handling (robust)
# -----------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))      # e.g. /path/to/project/api
PARENT_DIR = os.path.dirname(THIS_DIR)                     # e.g. /path/to/project
# search locations in order: same dir as app, then parent dir
SEARCH_DIRS = [THIS_DIR, PARENT_DIR]

def find_file_anywhere(filenames, search_dirs=SEARCH_DIRS):
    """Return first existing path for any filename in filenames (try search_dirs order)."""
    for d in search_dirs:
        for fname in filenames:
            p = os.path.join(d, fname)
            if os.path.exists(p):
                return p
    return None

# filenames you have in the repo (adjust names if different)
model_candidates = ["model (1).pkl", "model.pkl"]
scaler_candidates = ["standar.pkl", "standard.pkl", "standardscaler.pkl"]
minmax_candidates = ["minmaxscaler.pkl", "minmax_scaler.pkl", "minmax.pkl"]

model_path = find_file_anywhere(model_candidates)
scaler_path = find_file_anywhere(scaler_candidates)
minmax_path = find_file_anywhere(minmax_candidates)

missing = []
if model_path is None:
    missing.append(f"Model file not found (tried {model_candidates})")
if scaler_path is None:
    missing.append(f"Standard scaler file not found (tried {scaler_candidates})")
if minmax_path is None:
    missing.append(f"MinMax scaler file not found (tried {minmax_candidates})")

if missing:
    # raise an informative error early
    msg = " ; ".join(missing) + f".\nSearched in: {SEARCH_DIRS}"
    logger.error(msg)
    raise FileNotFoundError(msg)

logger.info(f"Using model: {model_path}")
logger.info(f"Using standard scaler: {scaler_path}")
logger.info(f"Using minmax scaler: {minmax_path}")

# -----------------------
# Load objects safely
# -----------------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_pickle(model_path)
sc = load_pickle(scaler_path)   # presumably StandardScaler
ms = load_pickle(minmax_path)   # presumably MinMaxScaler

# -----------------------
# Flask app (templates in parent dir)
# -----------------------
# find template folder (look in THIS_DIR/templates then PARENT_DIR/templates)
templates_dir = None
for d in SEARCH_DIRS:
    t = os.path.join(d, "templates")
    if os.path.isdir(t):
        templates_dir = t
        break

if templates_dir is None:
    raise FileNotFoundError(f"No 'templates' directory found in {SEARCH_DIRS}")

logger.info(f"Using templates folder: {templates_dir}")

app = Flask(__name__, template_folder=templates_dir)

# crop mapping — adjust if your model uses different label encoding
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
    6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon",
    10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
    17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
    20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Use .get so missing form keys don't KeyError
        N = float(request.form.get("Nitrogen", 0))
        P = float(request.form.get("Phosporus", 0))   # preserves your HTML name
        K = float(request.form.get("Potassium", 0))
        temp = float(request.form.get("Temperature", 0))
        humidity = float(request.form.get("Humidity", 0))
        ph = float(request.form.get("Ph", 0))
        rainfall = float(request.form.get("Rainfall", 0))

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Apply same transforms as during training (order matters)
        # Your code used ms.transform then sc.transform — keep same order here
        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)

        prediction = model.predict(final_features)
        pred = prediction[0]

        # handle different output types
        result_text = ""
        # If predicted a numeric label that matches crop_dict:
        try:
            # convert numpy.int64 etc. to int
            if isinstance(pred, (np.integer, int)):
                idx = int(pred)
                if idx in crop_dict:
                    result_text = f"{crop_dict[idx]} is the best crop to cultivate there."
                else:
                    result_text = f"Predicted label: {idx}"
            else:
                # probably a string label already
                result_text = f"{str(pred)} is the best crop to cultivate there."
        except Exception:
            result_text = f"Prediction: {str(pred)}"

        return render_template("index.html", result=result_text)

    except Exception as e:
        logger.exception("Prediction error")
        return render_template("index.html", result=f"Error: {str(e)}")

if __name__ == "__main__":
    # host=0.0.0.0 if you run in docker or remote server
    app.run(debug=True, host="127.0.0.1", port=5000)
