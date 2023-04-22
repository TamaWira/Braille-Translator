import os
import cv2
import sys

from helpers import BrailleConverter
from helpers.BrailleImage import BrailleImage
from helpers.BrailleClassifier import BrailleClassifier
from helpers.SegmentationEngine import SegmentationEngine
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

SERVING_ROOT = os.path.join(os.getcwd(), "static\serving")

ORIGINAL_IMAGE_FILENAME = "original_image.jpg"
ORIGINAL_IMAGE_ROOT = os.path.join(SERVING_ROOT, ORIGINAL_IMAGE_FILENAME)

CONVERTED_IMAGE_FILENAME = "converted_image.jpg"
CONVERTED_IMAGE_ROOT = os.path.join(SERVING_ROOT, CONVERTED_IMAGE_FILENAME)

CROPPED_IMAGE_FILENAME = "cropped_image.png"
CROPPED_IMAGE_ROOT = os.path.join(SERVING_ROOT, CROPPED_IMAGE_FILENAME)

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
    letters = []
    img = BrailleImage(image=ORIGINAL_IMAGE_ROOT)
    cv2.imwrite(CONVERTED_IMAGE_ROOT, img.get_converted_image())
    img = BrailleImage(image=CONVERTED_IMAGE_ROOT)
    classifier = BrailleClassifier()
    for letter in SegmentationEngine(img):
        letters.append(letter.get_bounding_box())
        letter.mark()
        classifier.push_and_predict(img, letter, classify=False)
    
    final_img = img.get_final_image()
    final_text = classifier.digest()

    print(final_text)

    try:
        if CROPPED_IMAGE_FILENAME in os.listdir(SERVING_ROOT):
            os.remove(CROPPED_IMAGE_ROOT)
        cv2.imwrite(CROPPED_IMAGE_ROOT, final_img)
    except:
        pass

    return render_template('result.html', predicted_result=final_text)

if __name__ == "__main__":
    app.run('localhost', 3001, debug=True)