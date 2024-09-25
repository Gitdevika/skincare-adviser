from flask import Flask, render_template, request, jsonify, url_for,redirect
import numpy as np
import cv2
import dlib
from tensorflow.keras.models import load_model
import pymysql


app = Flask(__name__, static_folder='templates')

# MySQL database connection
db = pymysql.connect(host='localhost', user='root', password='12345', database='mydatabase', charset='utf8mb4')
cursor = db.cursor()

# Load the three models
acne_model = load_model('acne_final_model.keras')
darkspots_model = load_model('darkspots_final_model.keras')
wrinkles_model = load_model('wrinkles_final_model.keras')

# Load the face detector from dlib
face_detector = dlib.get_frontal_face_detector()

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    return image

# Function to detect face in the image
def detect_face(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    return len(faces) > 0

# Function to predict using the models
def predict(image_path):
    if not detect_face(image_path):
        return {'error': 'Please upload an image containing a face'}
    
    image = preprocess_image(image_path)

    acne_prediction = acne_model.predict(np.expand_dims(image, axis=0))
    darkspots_prediction = darkspots_model.predict(np.expand_dims(image, axis=0))
    wrinkles_prediction = wrinkles_model.predict(np.expand_dims(image, axis=0))

    # Map predictions to actual severities
    acne_label = np.argmax(acne_prediction)  # Get index of maximum value (severity)
    darkspots_label = np.argmax(darkspots_prediction)
    wrinkles_label = np.argmax(wrinkles_prediction)

    severity_labels = {
        0: 'mild',
        1: 'moderate',
        2: 'severe'
    }

    acne_severity_real=severity_labels[acne_label]
    darkspots_severity_real=severity_labels[darkspots_label]
    wrinkles_severity_real=severity_labels[wrinkles_label]
    # Update severity values based on model predictions
    acne_severity = severity_labels[acne_label]
    if acne_severity in ['moderate', 'severe']:
        acne_severity = 'acne'
    darkspots_severity = severity_labels[darkspots_label]
    if darkspots_severity in ['moderate', 'severe']:
        darkspots_severity = 'darkspots'
    wrinkles_severity = severity_labels[wrinkles_label]
    if wrinkles_severity in ['moderate', 'severe']:
        wrinkles_severity = 'wrinkles'

    return {
        'acne1': acne_severity_real,
        'darkspots1': darkspots_severity_real,
        'wrinkles1': wrinkles_severity_real,
        'acne': acne_severity,
        'darkspots': darkspots_severity,
        'wrinkles': wrinkles_severity
    }


# Function to fetch product information from database
def fetch_product_info(age_group, gender, skin_type, acne_severity, darkspots_severity, wrinkles_severity):
    query = f"SELECT * FROM products WHERE age = '{age_group}' AND gender = '{gender}' AND skintype = '{skin_type}' AND  (skincondition= '{acne_severity}' OR skincondition= '{darkspots_severity}' OR skincondition= '{wrinkles_severity}')"
    print(query)  # Print the constructed query for debugging
    cursor.execute(query)
    results = cursor.fetchall()
    return results


@app.route('/')
@app.route('/landing')
def landing():
    return render_template('landing.html')

@app.route('/index')
def index():
    return render_template('index.html')
    

@app.route('/predict', methods=['POST'])
def handle_predict():
    if 'file' not in request.files:
        return redirect(url_for('index', error='No file part'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index', error='No selected file'))
    
      # Check if any form fields are empty
    required_fields = ['gender', 'skin_type', 'age_group']
    for field in required_fields:
        if not request.form.get(field):
            return redirect(url_for('index', error='Please fill all the fields'))

    file_path = 'temp.jpg'
    file.save(file_path)
    results = predict(file_path)

    if 'error' in results:
        return redirect(url_for('index', error=results['error']))

   # Fetch product information from database based on model predictions
    product_info = fetch_product_info(request.form['age_group'], request.form['gender'], request.form['skin_type'], results['acne'], results['darkspots'], results['wrinkles'])
    
    # Prepare the data to send back to frontend
    response = {
        'results': results,
        'product_info': product_info
    }
    return render_template('results.html', results=results, product_info=product_info)

if __name__ == '__main__':
    app.run(debug=True)