from utils.disease_info import get_disease_info
from flask import Flask, render_template, request
from utils.predictor import predict_disease
from utils.image_utils import save_uploaded_image
import os
from utils.helpers import (
    format_confidence,
    get_web_path
)
from utils.gradcam import generate_gradcam
import time
from utils.helpers import (
    format_confidence,
    get_web_path,
    get_severity_badge
)
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB
UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    image = request.files["image"]

    try:

        image_path = save_uploaded_image(
            image,
            app.config["UPLOAD_FOLDER"]
        )

    except ValueError as e:

        return str(e)
    start_time = time.perf_counter()
    prediction = predict_disease(image_path)
    gradcam_path = generate_gradcam(
    image_path
)
    info = get_disease_info(
    prediction["disease"]
)
    end_time = time.perf_counter()
    inference_time = round(
    end_time - start_time,
    3 
    )
    return render_template(

    "result.html",

    image_path=get_web_path(image_path),

    disease=info["name"],

    crop=info["crop"],

    severity=get_severity_badge(
    info["severity"]
    
    ),

    confidence=format_confidence(
    prediction["confidence"]
    
),

    symptoms=info["symptoms"],

    cause=info["cause"],

    treatment=info["treatment"],

    prevention=info["prevention"],

    organic_treatment=info["organic_treatment"],

    chemical_treatment=info["chemical_treatment"],

    top3_predictions=prediction["top3_predictions"],
    gradcam_path=gradcam_path,

    inference_time=inference_time,

model_name="ResNet50 Transfer Learning",

framework="TensorFlow 2.21",

num_classes=38,

input_size="224 × 224",

explainability="Grad-CAM",

)


if __name__ == "__main__":
    app.run(debug=True)