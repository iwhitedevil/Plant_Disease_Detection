import numpy as np
import cv2
import tensorflow as tf
from flask import Flask,render_template, request, send_from_directory
import os

app = Flask(__name__)

STATIC_FOLDER = 'static'
# Path to the folder where we'll store the upload before prediction
UPLOAD_FOLDER = STATIC_FOLDER + '/uploads'
# Path to the folder where we store the different models
MODEL_FOLDER = STATIC_FOLDER + '/models'

#Name of Classes
CLASS_NAMES = ['Corn-Common_rust', 'Tomato-Bacterial_spot', 'Potato-Early_blight']
def load__model():
    """Load model once at running time for all the predictions"""
    print('[INFO] : Model loading ................')
    tf.compat.v1.disable_eager_execution()
    model = tf.keras.models.load_model(MODEL_FOLDER + '/plant_disease.h5')
    global graph
    graph = tf.compat.v1.get_default_graph()
    print('[INFO] : Model loaded')
    return model


def Prediction(plant_image):
    model = load__model()
    # Open the image file
    with open(plant_image, 'rb') as f:
        # Convert image to NumPy array
        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        # Resizing the image
        opencv_image = cv2.resize(opencv_image, (256, 256))
        # Convert image to 4 Dimension
        opencv_image.shape = (1, 256, 256, 3)
        # Make Prediction

    with graph.as_default():
        result = model.predict(opencv_image)
        category = CLASS_NAMES[np.argmax(result)]
        confidence = round(100 * (np.max(result[0])), 2)
    return category,confidence

# Home Page
@app.route('/')
def index():
    return render_template('index.html')


# Process file and predict his label
@app.route('/upload', methods=['GET', 'POST'])

def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image']
        fullname = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(fullname)

        category, confidence = Prediction(fullname)

        return render_template('predict.html', image_file_name=file.filename, label=category, accuracy=confidence)


@app.route('/upload/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)




if __name__ == '__main__':
    app.run(debug=True)