import os


# ==========================================================
# Format Confidence
# ==========================================================

def format_confidence(confidence):
    """
    Convert confidence (0-1) into percentage string.
    """

    return f"{confidence * 100:.2f}"


# ==========================================================
# Convert Local Path to Web Path
# ==========================================================

def get_web_path(file_path):
    """
    Convert Windows/Linux path to browser-friendly path.
    """

    return "/" + file_path.replace("\\", "/")


# ==========================================================
# Format Class Name (Fallback)
# ==========================================================

def format_class_name(class_name):
    """
    Convert model class name into readable text.

    Example:
    Tomato___Late_blight
    ->
    Tomato Late Blight
    """

    return (
        class_name
        .replace("___", " ")
        .replace("_", " ")
    )


# ==========================================================
# Severity Color
# ==========================================================

def get_severity_color(severity):
    """
    Returns Bootstrap-like color names.
    """

    colors = {

        "None": "success",

        "Low": "info",

        "Medium": "warning",

        "High": "danger",

        "Very High": "dark"

    }

    return colors.get(
        severity,
        "secondary"
    )

def get_severity_badge(severity):

    badges = {
        "None": "🟢 Healthy",
        "Low": "🟢 Low",
        "Medium": "🟡 Medium",
        "High": "🟠 High",
        "Very High": "🔴 Very High"
    }

    return badges.get(severity, severity)