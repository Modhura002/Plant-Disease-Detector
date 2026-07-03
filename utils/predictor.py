import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_model.keras"
)

CLASS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "class_names.json"
)

# ==========================================================
# Load Model (Only Once)
# ==========================================================

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "preprocess_input": preprocess_input
    }
)

with open(CLASS_PATH, "r") as file:
    class_names = json.load(file)

print("✅ Model Loaded Successfully")


# ==========================================================
# Image Preprocessing
# ==========================================================

def preprocess_image(img_path):

    img = image.load_img(
        img_path,
        target_size=(224,224)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    

    return img_array


# ==========================================================
# Top 3 Predictions
# ==========================================================

def get_top_predictions(prediction):

    top3_indices = np.argsort(prediction)[-3:][::-1]

    results = []

    for idx in top3_indices:

        results.append({

            "disease": class_names[idx]
                .replace("___", " ")
    .replace("_", " "),

            "confidence": round(
                float(prediction[idx])*100,
                2
            )

        })

    return results


# ==========================================================
# Main Prediction Function
# ==========================================================

def predict_disease(img_path):

    img_array = preprocess_image(img_path)

    prediction = model.predict(
        img_array,
        verbose=0
    )[0]

    predicted_index = np.argmax(prediction)

    disease = class_names[predicted_index]

    confidence = float(prediction[predicted_index])

    top3_predictions = get_top_predictions(prediction)

    return {

        "disease": disease,

        "confidence": confidence,

        "top3_predictions": top3_predictions

    }