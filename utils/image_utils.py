import os
import uuid
from PIL import Image


# ==========================================================
# Allowed Image Extensions
# ==========================================================

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp"
}


# ==========================================================
# Check File Extension
# ==========================================================

def allowed_file(filename):
    """
    Check if uploaded file has a valid image extension.
    """

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ==========================================================
# Verify Image
# ==========================================================

def verify_image(file):
    """
    Verify that the uploaded file is a valid image.
    """

    try:

        img = Image.open(file)
        img.verify()

        # Reset file pointer after verification
        file.seek(0)

        return True

    except Exception:

        return False


# ==========================================================
# Generate Unique Filename
# ==========================================================

def generate_filename(filename):
    """
    Generate a unique filename while preserving extension.
    """

    extension = filename.rsplit(".", 1)[1].lower()

    return f"{uuid.uuid4().hex}.{extension}"


# ==========================================================
# Save Uploaded Image
# ==========================================================

def save_uploaded_image(file, upload_folder):
    """
    Validate and save uploaded image.

    Parameters
    ----------
    file : FileStorage
        Uploaded Flask file object

    upload_folder : str
        Folder where image will be stored

    Returns
    -------
    str
        Full saved image path

    Raises
    ------
    ValueError
        If uploaded file is invalid.
    """

    # Empty filename
    if file.filename == "":
        raise ValueError("No file selected.")

    # Invalid extension
    if not allowed_file(file.filename):
        raise ValueError(
            "Only JPG, JPEG, PNG and WEBP images are allowed."
        )

    # Invalid image
    if not verify_image(file):
        raise ValueError(
            "Uploaded file is not a valid image."
        )

    # Create upload folder if it doesn't exist
    os.makedirs(upload_folder, exist_ok=True)

    # Generate unique filename
    filename = generate_filename(file.filename)

    image_path = os.path.join(
        upload_folder,
        filename
    )

    # Save image
    file.save(image_path)

    return image_path