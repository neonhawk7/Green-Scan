from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json
import os
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # ✅ Enables CORS for frontend communication

# ✅ Set Correct Paths
MODEL_PATH = os.getenv("MODEL_PATH", r"D:\Plant Disease Detection Using AI\backend\models\resnet18_trained_v2.pth")
CLASS_LABELS_PATH = os.getenv("CLASS_LABELS_PATH", r"D:\Plant Disease Detection Using AI\backend\models\class_labels_v2.json")

# ✅ Ensure files exist before proceeding
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Error: Model file is missing at {MODEL_PATH}")

if not os.path.exists(CLASS_LABELS_PATH):
    raise FileNotFoundError(f"❌ Error: Class labels file is missing at {CLASS_LABELS_PATH}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load class labels with treatments from `class_labels_v2.json`
with open(CLASS_LABELS_PATH, "r") as f:
    raw_labels = json.load(f)

# ✅ Ensure all indexes exist
class_labels = []
for i in range(len(raw_labels)):
    if str(i) in raw_labels:
        class_labels.append(raw_labels[str(i)]["disease"])
    else:
        print(f"⚠ Warning: Missing key '{i}' in class_labels_v2.json")

def get_treatment(disease_name):
    """Fetch treatment from class_labels_v2.json"""
    index = class_labels.index(disease_name) if disease_name in class_labels else -1
    return raw_labels.get(str(index), {}).get("treatment", "No specific cure available.")

num_classes = len(class_labels)

# ✅ Define image transformations (Move this transformation to a separate function for reuse)
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0).to(device)

# ✅ Load Model
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["image"]
        image = Image.open(file).convert("RGB")
        
        # ✅ Preprocess the image (resize, normalize, convert to tensor)
        image_tensor = preprocess_image(image)

        # ✅ Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        # ✅ Debugging Logs
        print("\n🔥 DEBUGGING PREDICTION 🔥")
        print(f"📊 Raw Model Outputs: {outputs.tolist()}")
        print(f"📊 Softmax Probabilities: {probabilities.tolist()}")
        print(f"🔢 Predicted Class Index: {predicted.item()}")
        print(f"✅ Disease Name: {class_labels[predicted.item()]}")
        print(f"⚡ Confidence Score: {confidence.item():.2%}")

        # ✅ Get disease name & treatment
        predicted_class = class_labels[predicted.item()]
        cure_solution = get_treatment(predicted_class)

        return jsonify({
            "disease": predicted_class,
            "confidence": round(confidence.item() * 100, 2),
            "treatment": cure_solution
        })

    except Exception as e:
        print(f"❌ Error: {str(e)}")  # Log error to console
        return jsonify({"error": "An internal error occurred. Please try again."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
