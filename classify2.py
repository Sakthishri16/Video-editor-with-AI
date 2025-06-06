import json
import joblib
import os
import numpy as np

def classify():
    model_path = "svm_model.pkl"
    vectorizer_path = "tfidf_vectorizer.pkl"
    json_file = "output.json"

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("SVM model or TF-IDF vectorizer not found!")

    print("🔄 Loading model and vectorizer...")
    svm_model = joblib.load(model_path)
    tfidf_vectorizer = joblib.load(vectorizer_path)

    with open(json_file, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "transcription" in data:
        speech_segments = [seg for seg in data["transcription"] if seg["type"] == "speech"]
    elif isinstance(data, list):
        speech_segments = [seg for seg in data if seg["type"] == "speech"]
    else:
        raise ValueError("Unexpected JSON structure!")

    texts = [seg["text"] for seg in speech_segments]

    if texts:
        print("🔄 Transforming text with TF-IDF vectorizer...")
        text_features = tfidf_vectorizer.transform(texts)

        predictions = svm_model.predict(text_features)

        print("📝 Updating JSON file with classifications...")
        for segment, label in zip(speech_segments, predictions):
            segment["classification"] = "Relevant" if label == 1 else "Irrelevant"

        with open(json_file, "w") as f:
            json.dump(data, f, indent=4)

        print(f"✅ Classification completed and saved in {json_file}")
    else:
        print("⚠️ No speech segments found for classification.")

# Ensure it only runs when explicitly called
if __name__ == "__main__":
    classify()
