import os, json, re
import numpy as np
from PIL import Image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
import google.generativeai as genai

# ==== Wajib isi API key via environment, biar aman ====
# Windows (PowerShell):  setx GEMINI_API_KEY "YOUR_KEY"
# macOS/Linux (bash):    export GEMINI_API_KEY="YOUR_KEY"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY belum di-set. Set dulu di environment ya.")

# Konfigurasi Gemini
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.5-pro"   

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads/"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Muat model klasifikasi
MODEL_PATH = "oral_diseases_model_tuned.h5"
CLASS_NAMES = ['Calculus', 'Gingivitis', 'Mouth Ulcer', 'Tooth Discoloration', 'hypodontia']
model = tf.keras.models.load_model(MODEL_PATH)

# ===== Utils =====
def preprocess_image(image_path, target_size=(128, 128)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def ask_gemini(disease_name: str) -> dict:
    """
    Minta Gemini balikin JSON berisi:
    {
      "severity": "Ringan|Sedang|Serius",
      "description": "teks 1–2 kalimat",
      "recommendations": ["...", "...", "...", "...", "..."]
    }
    """
    prompt = f"""
Kamu adalah dokter gigi. Untuk penyakit gigi: "{disease_name}",
berikan output dalam **JSON valid** TANPA teks lain, format persis:
{{
  "severity": "Ringan|Sedang|Serius",
  "description": "<1-2 kalimat, bahasa Indonesia, ringkas>",
  "recommendations": [
    "<5 butir rekomendasi praktis, kalimat singkat>",
    "<...>",
    "<...>",
    "<...>",
    "<...>"
  ]
}}

Hindari kata pembuka/penutup, jangan pakai markdown, balikkan HANYA JSON.
"""
    model_g = genai.GenerativeModel(GEMINI_MODEL)
    resp = model_g.generate_content(prompt)
    text = (resp.text or "").strip()

    # Ambil blok JSON murni (kalau Gemini nakal & kasih teks tambahan)
    match = re.search(r"\{.*\}", text, flags=re.S)
    json_str = match.group(0) if match else text

    data = json.loads(json_str)
    # Validasi minimal
    if not isinstance(data.get("recommendations", []), list) or not data.get("description"):
        raise ValueError("Response JSON tidak sesuai skema.")
    # Keep max 5 rekomendasi
    data["recommendations"] = data["recommendations"][:5]
    return data

# ===== Routes =====
@app.route("/", methods=["GET", "POST"])
def index():
    ctx = {
        "filename": None,
        "prediction": None,
        "confidence": None,
        "severity": None,
        "description": None,
        "reco_list": [],
        "err_msg": None,
    }

    if request.method == "POST":
        if "file" not in request.files:
            ctx["err_msg"] = "Tidak ada file pada form."
            return render_template("index.html", **ctx)

        file = request.files["file"]
        if not file or file.filename == "":
            ctx["err_msg"] = "Silakan pilih gambar terlebih dulu."
            return render_template("index.html", **ctx)

        filename = secure_filename(file.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(path)
        ctx["filename"] = filename

        # Prediksi kelas
        x = preprocess_image(path)
        preds = model.predict(x)
        idx = int(np.argmax(preds[0]))
        prob = float(preds[0][idx])
        label = CLASS_NAMES[idx]

        ctx["prediction"] = label
        ctx["confidence"] = round(prob * 100, 2)

        # Minta rekomendasi & deskripsi dari Gemini SAJA
        try:
            g = ask_gemini(label)
            ctx["severity"] = g.get("severity", "Sedang")
            ctx["description"] = g.get("description", "")
            ctx["reco_list"] = g.get("recommendations", [])
        except Exception as e:
            ctx["err_msg"] = f"Gagal mengambil rekomendasi dari Gemini: {e}"

    return render_template("index.html", **ctx)

if __name__ == "__main__":
    app.run(debug=True)
