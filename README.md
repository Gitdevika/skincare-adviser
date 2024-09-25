# Skincare Advisor

A web application that utilizes machine learning to analyze skin conditions from uploaded photos and user-given details(age ,gender,skintype)and provides personalized skincare product recommendations based on the user's profile and detected skin issues.

## Features

- **Skin Analysis**: Upload a photo for analysis of acne, dark spots, and wrinkles.
- **Severity Detection**: Classifies skin conditions as mild, moderate, or severe.
- **Product Recommendations**: Suggests skincare products based on user profile and conditions.
- **User-Friendly Interface**: Intuitive design for easy navigation and photo uploads.
- **Face Detection**: Ensures uploaded images contain a face for accurate analysis.
- **Data Augmentation**: Enhances model performance through varied training data.
- **Targeted Model Training**: Separate models for acne, dark spots, and wrinkles.
- **Responsive Design**: Optimized for desktop, tablet, and mobile use.

## Technologies Used

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask
- **Machine Learning**: TensorFlow, Keras
- **Image Processing**: OpenCV, dlib
- **Database**: MySQL
- **Data Handling**: Pandas (for data manipulation)
- **Environment**: Python

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Gitdevika/skincare-adviser.git
   ```

2. **Install required dependencies**:
   Make sure to install the necessary packages. You can do this using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the model training scripts**:
   Before running the Flask app, you need to create the three models. So run:
   ```bash
      threemodel.py
   ```
   This will generate the required model files.

4. **Set up the database**:
   The product recommendation database is in Excel format. To use it with the Flask app, you need to store it in MySQL. Please ensure that the database structure aligns with the queries in the Flask code.

5. **Run the Flask app**:
   After setting up the models and database, you can run the Flask application:
   ```bash
   python app.py
   ```

6. **Access the application**:
   Open your web browser and go to `http://127.0.0.1:5000/` to access the skincare advisor website.

## Project Structure

- `dataset/` - Contains the training and testing datasets.
-  threemodel.py - Code to Create three models.
-  app.py - Flask application .
- `templates/` - HTML files for the web interface and the folder in which the product images are in .
- products - Excel sheet of the product details.\



## Team Members

- Devika Sreenivasan K
- Asha K Wilson 
- Sreelakshmi E
- Angel P Shaji
