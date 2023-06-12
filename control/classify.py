import os
import cv2
import PIL
import json
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import load_model
from .segmentation import BrailleSegmentation
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore

class BrailleClassifier:
    def __init__(self,
                 model_path=os.path.abspath("weights/cnn_v1.hdf5"), 
                 json_path="utils/class_labels.json",
                 symbols_path="utils/braille_symbols.json"):
        
        self.segmenter = BrailleSegmentation()
        self.dim = (70, 100)
        self.json_path = json_path
        self.model = load_model(model_path)
        self.symbols_path = symbols_path
        self.output_path = "outputs/output.png"

    def import_class_file(self):
        """Retrieve class labels from JSON file"""

        with open(self.json_path) as json_file:
            return json.load(json_file)
    
    def get_class(self, prediction, class_labels):
        """Convert from index into text label"""

        for key, value in class_labels.items():
            if prediction == value:
                return key
            
    def convert_symbols(self, symbols):
        """Convert text into symbols"""

        with open(self.symbols_path) as path:
            braille_symbols = json.load(path)

        try:
            return braille_symbols[symbols]
        except:
            return symbols
        
    def draw_text(self, image, text, org):
        cv2.putText(
            img=image,
            text=text,
            org=org,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0,0,0),
            thickness=2,
            lineType=cv2.LINE_AA)
    
    def preprocess_cells(self, cell):
        """Reshape cropped braille cells to desired dimension"""

        braille_letter = PIL.Image.fromarray(cell)
        processed_img = braille_letter.resize(self.dim)
        processed_img = img_to_array(processed_img)
        processed_img = processed_img / 255.0
        processed_img = processed_img.reshape(1, 100, 70, 3)

        return processed_img
    
    def convert_coordinate(self, box):
        """Get start and end point from center, width, and height"""

        xcent, ycent, w, h = box[:4].astype(int)

        top = ycent - int(h/2) - 5 # Minus 5 for broader bbox
        left = xcent - int(w/2) - 5
        bottom = ycent + int(h/2) + 5
        right = xcent + int(w/2) + 5

        start_point = (left, top)
        end_point = (right, bottom)

        return start_point, end_point, xcent, ycent
    
    def convert_numbers(self, text):
        with open("utils/braille_numbers.json") as data:
            numbers_dict = json.load(data)
        
        try:
            return numbers_dict[text]
        except:
            return text
    
    def classify_braille(self, model, image, class_labels):
        label = model.predict(image, verbose=0)
        label = label.argmax(axis=-1)[0]
        label = self.get_class(label, class_labels)
        label = self.convert_symbols(label)

        return label
    
    def get_raw_texts(self, boxes, image, model, class_labels):
        texts = []

        for row in boxes:
            row_text = []
            for box in row:
                start_point, end_point, _, _ = self.convert_coordinate(box)

                # Get cropped boxes
                x1, y1 = start_point
                x2, y2 = end_point
                cropped_image = image[y1:y2, x1:x2]
                cropped_image = self.preprocess_cells(cropped_image)

                # Predict
                label = self.classify_braille(model, cropped_image, class_labels)
                row_text.append(label)
            
            texts.append(row_text)
        
        return texts
    
    def get_spaces(self, boxes, raw_texts):
        for i, row in enumerate(boxes):
            space_index = []

            # Get space
            xs, _, distances, common = self.segmenter.get_box_properties(row)
            for j in range(len(raw_texts[i])):
                if j > 0:
                    if distances[j-1] > common*1.5:
                        space_index.append(j)
            
            space_index.reverse()
            for j in space_index:
                raw_texts[i].insert(j, ' ')
        
        return raw_texts
    
    def join_texts(self, spaced_texts):
        joined_texts = [''.join(sentences).split(' ') for sentences in spaced_texts]
        joined_texts = [[*teks] for teks in joined_texts]

        for i, sents in enumerate(joined_texts):
            sents = [sent.replace('^', ' ') for sent in sents]
            joined_texts[i] = sents
            for j, sent in enumerate(sents):
                if sent[0] == '#':
                    sent = [*sent]
                    sent[0] = ' '
                    nums = []
                    for _, letter in enumerate(sent):
                        nums.append(self.convert_numbers(letter))
                    joined_texts[i][j] = ''.join(nums)
        
        return joined_texts
    
    def draw_final(self, image, boxes, joined_texts):
        for i, row in enumerate(boxes):
            teks = [''.join(joined_texts[i])][0]
            xs, _, distances, common = self.segmenter.get_box_properties(row)
            for i, box in enumerate(row):
                start_point, end_point, xcent, ycent = self.convert_coordinate(box)
                
                # Draw result
                self.draw_text(image, teks[i], (xcent-10, ycent+30))
                cv2.rectangle(image, start_point, end_point, (0,0,0), 1)

        return image
    
    def recognize_braille(self, image_path):
        image = cv2.imread(image_path)
        class_labels = self.import_class_file()

        _, list_boxes = self.segmenter.segment_braille(image_path)
        boxes = self.segmenter.clean_bboxes(list_boxes)
        raw_texts = self.get_raw_texts(boxes, image, self.model, class_labels)
        spaces_texts = self.get_spaces(boxes, raw_texts)
        joined_texts = self.join_texts(spaces_texts)

        final_image = self.draw_final(image, boxes, joined_texts)

        return final_image

    def translate_braille(self, image_path, method='cnn', show=True):
        """Recognize Braille Letters inside an Image"""

        # Pre-requisites
        text = []
        number = []
        image = cv2.imread(image_path)
        drawed_image = image.copy()
        if method == 'cnn':
            class_labels = self.import_class_file()
            model = load_model(self.model_path)

        # Get boxes
        segmentation_engine = BrailleSegmentation()
        _, boxes = segmentation_engine.segment_braille(image_path)
        boxes = segmentation_engine.clean_bboxes(boxes)
        
        # Draw boxes
        for row in boxes:
            for box in row:
                start_point, end_point, xcent, ycent = self.convert_coordinate(box)
                if method == 'yolo':
                    label = self.convert_yolo_class(box)
                elif method == 'cnn':

                    # Get Cropped Boxes
                    x1, y1 = start_point
                    x2, y2 = end_point
                    cropped_image = image[y1:y2, x1:x2]
                    cropped_image = self.preprocess_cells(cropped_image)

                    # Prediction
                    label = model.predict(cropped_image, verbose=0)
                    label = label.argmax(axis=-1)[0]
                    label = self.get_class(label, class_labels)
                    # text.append(label)
                
                label = self.convert_symbols(label)
                self.draw_text(drawed_image, label, (xcent-10, ycent+30))
                cv2.rectangle(drawed_image, start_point, end_point, (0,0,0), 1)
        
        if show:
            plt.figure(figsize=(10,7))
            plt.imshow(drawed_image)
            plt.axis('off')
            plt.tight_layout()
            plt.show()

        return drawed_image
