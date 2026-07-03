import json
import os

# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to JSON database
JSON_PATH = os.path.join(
    BASE_DIR,
    "data",
    "disease_info.json"
)

# Load disease database only once
with open(JSON_PATH, "r", encoding="utf-8") as file:
    disease_database = json.load(file)


def get_disease_info(disease_name):
    """
    Returns complete disease information for a predicted class.
    """

    return disease_database.get(
        disease_name,
        {
            "name": disease_name,
            "crop": "Unknown",
            "severity": "Unknown",
            "symptoms": [],
            "cause": "Information unavailable.",
            "treatment": [],
            "prevention": [],
            "organic_treatment": [],
            "chemical_treatment": []
        }
    )