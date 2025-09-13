import numpy as np
from densenet169 import densenet169_model  # Assuming you have your model defined in densenet_model.py
from PIL import Image
import os

if __name__ == '__main__':
    # Assuming you have test data, replace this with your actual test data loading function
    data_folder1 = "/Users/sohamchongder/Desktop/Medical Imaging project/cnn_finetune/test"
    img = Image.open(os.path.join(data_folder1, "test3.png"))
    img = img.convert("RGB")
    X_test=np.array(img.resize((224, 224)))
    X_test = np.expand_dims(X_test, axis=0)
    # print(X_test.shape)
    model = densenet169_model(img_rows=224, img_cols=224, color_type=3, num_classes=2)

    # Load the saved weights
    model.load_weights('weights/weights.h5')  # Assuming the weights are saved in weights/weights.h5

    # Perform inference on the test data
    predictions = model.predict(X_test)

    answers=["Covid negative","Covid positive"]
    # Assuming you want to print or use the predictions further
    print(predictions)
    print(answers[np.argmax(predictions)])