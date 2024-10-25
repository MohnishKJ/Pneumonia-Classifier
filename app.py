import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)
model = load_model('models/final_model.h5')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_health_tips(is_healthy):
    if is_healthy:
        return [
            "✓ Maintain a balanced diet rich in fruits, vegetables, and whole grains",
            "✓ Regular exercise helps keep your lungs strong and healthy",
            "✓ Stay up to date with vaccinations and preventive care"
        ]
    else:
        return [
            "⚠ Seek immediate medical attention for proper diagnosis and treatment",
            "⚠ Take prescribed antibiotics as directed by your healthcare provider",
            "⚠ Get plenty of rest and stay well-hydrated"
        ]

def get_recommendations(is_healthy):
    if is_healthy:
        return ("Your lungs appear healthy! Here are some tips to maintain your lung health:", 
                "Healthy Living Tips")
    else:
        return ("Signs of pneumonia detected. Please follow these important steps:", 
                "Important Action Steps")

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', 
                             error="No file uploaded",
                             show_error=True)

    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', 
                             error="No file selected",
                             show_error=True)

    if not allowed_file(file.filename):
        return render_template('index.html', 
                             error="Invalid file type. Please upload an image file (png, jpg, jpeg, gif).",
                             show_error=True)

    try:
        # Save the uploaded file temporarily
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # Process image
        img = load_img(file_path, target_size=(128, 128))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Make prediction
        prediction = model.predict(img_array)
        
        # Clean up
        os.remove(file_path)

        # Process results
        is_healthy = prediction[0][0] > 0.5
        result = "No Pneumonia Detected" if is_healthy else "Pneumonia Detected"
        color = "green" if is_healthy else "red"
        message, tips_title = get_recommendations(is_healthy)
        tips = get_health_tips(is_healthy)
        
        return render_template('index.html', 
                             prediction=result,
                             color=color,
                             message=message,
                             tips_title=tips_title,
                             tips=tips,
                             show_result=True)
    
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return render_template('index.html', 
                             error=f"Error processing the image: {str(e)}",
                             show_error=True)

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)