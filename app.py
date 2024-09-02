from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import torch
from torchvision import transforms
import timm
import logging
import base64
from io import BytesIO
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Setup CORS to allow requests from specific origins
CORS(app, resources={r"/*": {"origins": "*"}})

# Function to load the model


def load_model(model_name, num_classes, device, model_path):
    model = timm.create_model(
        model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Image preprocessing function


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Function to handle preflight CORS requests


def _build_cors_preflight_response():
    response = jsonify({'status': 'OK'})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers",
                         "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods",
                         "GET,PUT,POST,DELETE,OPTIONS")
    return response

# Kidney disease prediction route


@app.route('/predict_kidney_disease', methods=['POST', 'OPTIONS'])
@cross_origin(origins="http://localhost:3000")
def predict_kidney_image():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()

    try:
        data = request.json
        image_data = data['image']

        # Replace with your default model name
        model_name = data.get('model_name', 'efficientvit_m2')

        # Decode the image
        image = Image.open(
            BytesIO(base64.b64decode(image_data))).convert('RGB')

        # Preprocess the image
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        processed_image = preprocess_image(image).to(device)

        # Load the model
        num_classes = 4
        model = load_model(model_name, num_classes, device,
                           model_path='best_model/efficientvit_m2_kidney_disease_classifier.pth')

        classes = ['Cyst', 'Tumor', 'Stone', 'Normal']

        # Predict
        with torch.no_grad():
            outputs = model(processed_image)
            _, predicted = torch.max(outputs, 1)
            prediction = predicted.item()

        # Return the prediction result
        return jsonify({"prediction": classes[prediction]}), 200

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Oral disease prediction route


@app.route('/predict_oral_disease', methods=['POST', 'OPTIONS'])
@cross_origin(origins="http://localhost:3000")
def predict_oral_image():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()

    try:
        data = request.json
        image_data = data['image']

        # Replace with your default model name
        model_name = data.get('model_name', 'efficientvit_b0')

        # Decode the image
        image = Image.open(
            BytesIO(base64.b64decode(image_data))).convert('RGB')

        # Preprocess the image
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        processed_image = preprocess_image(image).to(device)

        # Load the model
        num_classes = 6
        model = load_model(model_name, num_classes, device,
                           model_path=f'best_model/{model_name}_oral_disease_classifier.pth')

        classes = ['Calculus', 'Caries', 'Gingivitis',
                   'Hypodontia', 'Tooth Discoloration', 'Ulcers']

        # Predict
        with torch.no_grad():
            outputs = model(processed_image)
            _, predicted = torch.max(outputs, 1)
            prediction = predicted.item()

        # Return the prediction result
        return jsonify({"prediction": classes[prediction]}), 200

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Cancer disease prediction route


@app.route('/predict_cancer_disease', methods=['POST', 'OPTIONS'])
@cross_origin(origins="http://localhost:3000")
def predict_cancer_image():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()

    try:
        data = request.json
        image_data = data['image']

        # Replace with your default model name
        model_name = data.get('model_name', 'mobilevitv2_100.cvnets_in1k')

        # Decode the image
        image = Image.open(
            BytesIO(base64.b64decode(image_data))).convert('RGB')

        # Preprocess the image
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        processed_image = preprocess_image(image).to(device)

        # Load the model
        num_classes = 4
        model = load_model(model_name, num_classes, device,
                           model_path='best_model/mobilevitv2_100.pth')

        classes = ['adenocarcinoma', 'large.cell.carcinoma',
                   'normal', 'squamous.cell.carcinoma']

        # Predict
        with torch.no_grad():
            outputs = model(processed_image)
            _, predicted = torch.max(outputs, 1)
            prediction = predicted.item()

        # Return the prediction result
        return jsonify({"prediction": classes[prediction]}), 200

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Index route for testing


@app.route('/', methods=['GET'])
@cross_origin(origins="http://localhost:3000")
def index():
    return "<h1>Disease Detector Backend</h1>"


# Run the app
if __name__ == '__main__':
    app.run(debug=False)
