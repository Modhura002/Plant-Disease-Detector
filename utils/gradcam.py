import os
import cv2
import numpy as np
import tensorflow as tf
import uuid
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from utils.helpers import get_web_path

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_model.keras"
)

OUTPUT_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "gradcam"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==========================================================
# Load Model
# ==========================================================

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "preprocess_input": preprocess_input
    }
)

base_model = model.get_layer("resnet50")

last_conv_layer = base_model.get_layer(
    "conv5_block3_out"
)

print("✅ GradCAM Ready")
def load_image(image_path):

    img = image.load_img(
        image_path,
        target_size=(224,224)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array

# ==========================================================
# Compute Gradients
# ==========================================================

def compute_gradients(img_array):

    with tf.GradientTape() as tape:

        # Feature maps from ResNet50
        conv_output = base_model(
            img_array,
            training=False
        )

        tape.watch(conv_output)

        # Remaining layers of your model
        x = model.get_layer(
            "global_average_pooling2d_1"
        )(conv_output)

        x = model.get_layer(
            "dense_2"
        )(x)

        x = model.get_layer(
            "dropout_1"
        )(x)

        predictions = model.get_layer(
            "dense_3"
        )(x)

        predicted_class = tf.argmax(
            predictions[0]
        )

        class_score = predictions[
            :,
            predicted_class
        ]

    grads = tape.gradient(
        class_score,
        conv_output
    )

    return (
        conv_output,
        grads,
        predicted_class
    )

# ==========================================================
# Generate Heatmap
# ==========================================================

def generate_heatmap(conv_output, grads):

    # Average gradients over spatial dimensions
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    # Remove batch dimension
    conv_output = conv_output[0]

    # Weighted sum of feature maps
    heatmap = tf.reduce_sum(
        conv_output * pooled_grads,
        axis=-1
    )

    # Keep only positive influences
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize
    heatmap /= (
        tf.reduce_max(heatmap) + 1e-10
    )

    return heatmap.numpy()

# ==========================================================
# Create Overlay
# ==========================================================

def create_overlay(img_array, heatmap):

    # Remove weak activations
    heatmap[heatmap < 0.30] = 0

    # Resize heatmap
    heatmap = cv2.resize(
        heatmap,
        (224, 224)
    )

    # Convert to 0-255
    heatmap = np.uint8(
        255 * heatmap
    )

    # Apply color map
    heatmap_color = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    # Original image
    original = img_array[0].astype(np.uint8)

    # Overlay
    overlay = cv2.addWeighted(
        original,
        0.6,
        heatmap_color,
        0.4,
        0
    )

    return overlay

# ==========================================================
# Save GradCAM Image
# ==========================================================

def save_gradcam(overlay):

    filename = f"{uuid.uuid4().hex}.png"

    save_path = os.path.join(
        OUTPUT_FOLDER,
        filename
    )

    cv2.imwrite(save_path, overlay)

    # Return URL, not filesystem path
    return f"/static/gradcam/{filename}"


# ==========================================================
# Main GradCAM Function
# ==========================================================

def generate_gradcam(image_path):

    # Load image
    img_array = load_image(image_path)

    # Compute gradients
    conv_output, grads, predicted_class = compute_gradients(
        img_array
    )

    # Generate heatmap
    heatmap = generate_heatmap(
        conv_output,
        grads
    )

    # Create overlay
    overlay = create_overlay(
        img_array,
        heatmap
    )

    # Save overlay
    overlay_path = save_gradcam(
        overlay
    )

    return overlay_path