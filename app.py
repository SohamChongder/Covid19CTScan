from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
from densenet169 import densenet169_model 
from resnet_50 import resnet50_model

app = Flask(__name__)

# model = densenet169_model(img_rows=224, img_cols=224, color_type=3, num_classes=2)
# model.load_weights('weights/weights.h5')

model1=resnet50_model(224,224,3,2)
model1.load_weights('weights/resnet50_weights.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Receive image file from frontend
    file = request.files['file']
    img = Image.open(file)
    img = img.convert("RGB")
    img = img.resize((224, 224))  # Resize the image
    X_test = np.array(img)  # Convert image to numpy array
    X_test = np.expand_dims(X_test, axis=0)  # Add batch dimension

    # Perform inference
    predictions = model1.predict(X_test)

    # Get prediction label
    labels = ["Covid negative", "Covid positive"]
    prediction_label = labels[np.argmax(predictions)]

    # Return prediction to frontend
    return jsonify({'prediction': prediction_label})

if __name__ == '__main__':
    app.run(debug=True)
