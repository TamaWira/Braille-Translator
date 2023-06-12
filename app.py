import os
import cv2
import sys

from control.classify import BrailleClassifier as classifier
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

SERVING_ROOT = os.path.join(os.getcwd(), "static\serving")

ORIGINAL_IMAGE_FILENAME = "original_image.jpg"
ORIGINAL_IMAGE_ROOT = os.path.join(SERVING_ROOT, ORIGINAL_IMAGE_FILENAME)

DETECTED_IMAGE_FILENAME = "detected_image.jpg"
DETECTED_IMAGE_ROOT = os.path.join(SERVING_ROOT, DETECTED_IMAGE_FILENAME)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    if request.method == 'POST':
        if request.files['file'] != '':
            if ORIGINAL_IMAGE_FILENAME in os.listdir(SERVING_ROOT):
                os.remove(ORIGINAL_IMAGE_ROOT)
            uploaded_file = request.files['file']
            uploaded_file.save(ORIGINAL_IMAGE_ROOT)
            return redirect(url_for('result'))
        else: 
            return redirect(url_for('predict'))

@app.route('/result')
def result():
    
    predicted_image = classifier().recognize_braille(ORIGINAL_IMAGE_ROOT)
    cv2.imwrite(DETECTED_IMAGE_ROOT, predicted_image)

    return render_template('result.html')

if __name__ == "__main__":
    app.run('localhost', 3001, debug=True)