import json
import cv2
import tensorflow as tf

from math import sqrt
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore

def get_distance(p1, p2):
        x1,y1 = p1
        x2,y2 = p2
        return ((x2 - x1)**2) + ((y2 - y1)**2)

def get_left_nearest(dots, diameter, left):
        nearest = None
        for dot in dots:
            x,y = dot[0]
            dist = int(x - left)
            if dist <= diameter:
                if nearest is None:
                    nearest = dot
                else:
                    X,Y = nearest[0]
                    DIST = int(X - left)
                    if DIST > dist:
                        nearest = dot
        return nearest

def get_right_nearest(dots, diameter, right):
        nearest = None
        for dot in dots:
            x,y = dot[0]
            dist = int(right - x)
            if dist <= diameter:
                if nearest is None:
                    nearest = dot
                else:
                    X,Y = nearest[0]
                    DIST = int(right - X)
                    if DIST > dist:
                        nearest = dot
        return nearest

def get_dot_nearest(dots, diameter, pt1):
        nearest = None
        diameter **= 2
        for dot in dots:
            point = dot[0]
            dist_from_pt1 = get_distance(point, pt1)
            if dist_from_pt1 <= diameter:
                if nearest is None:
                    nearest = dot
                else:
                    pt = nearest[0]
                    ndist_from_pt1 = get_distance(pt, pt1)
                    if ndist_from_pt1 >= dist_from_pt1:
                        nearest = dot
        return nearest



def get_combination(box, dots, diameter):
        result = [0,0,0,0,0,0]
        left,right,top,bottom = box

        midpointY = int((bottom - top)/2)
        end = (right, midpointY)
        start = (left, midpointY)
        width = int(right - left)

        corners = { (left,top): 1, (right,top): 4, (left, bottom): 3, (right,bottom): 6,
                (left): 2, (right): 5}

        for corner in corners:
            if corner != left and corner != right:
                D = get_dot_nearest(dots, int(diameter), corner)
            else:
                if corner == left:
                    D = get_left_nearest(dots, int(diameter), left)
                else:
                    D = get_right_nearest(dots, int(diameter), right)

            if D is not None:
                dots.remove(D)
                result[corners[corner]-1] = 1

            if len(dots) == 0:
                break
        return end,start,width,tuple(result);

def translate_to_number(value):
    if value == 'a':
        return '1'
    elif value == 'b':
        return '2'
    elif value == 'c':
        return '3'
    elif value == 'd':
        return '4'
    elif value == 'e':
        return '5'
    elif value == 'f':
        return '6'
    elif value == 'g':
        return '7'
    elif value == 'h':
        return '8'
    elif value == 'i':
        return '9'
    else:
        return '0'

class Symbol(object):
    def __init__(self, value = None, letter = False, special = False):
        self.is_letter = letter
        self.is_special = special
        self.value = value

    def is_valid(self):
        r = True
        r = r and (self.value is not None)
        r = r and (self.is_letter is not None or self.is_special is not None)
        return r

    def letter(self):
        return self.is_letter

    def special(self):
        return self.is_special

class BrailleClassifier(object):
    symbol_table = {
         (1,0,0,0,0,0): Symbol('a',letter=True),
         (1,1,0,0,0,0): Symbol('b',letter=True),
         (1,0,0,1,0,0): Symbol('c',letter=True),
         (1,0,0,1,1,0): Symbol('d',letter=True),
         (1,0,0,0,1,0): Symbol('e',letter=True),
         (1,1,0,1,0,0): Symbol('f',letter=True),
         (1,1,0,1,1,0): Symbol('g',letter=True),
         (1,1,0,0,1,0): Symbol('h',letter=True),
         (0,1,0,1,0,0): Symbol('i',letter=True),
         (0,1,0,1,1,0): Symbol('j',letter=True),
         (1,0,1,0,0,0): Symbol('K',letter=True),
         (1,1,1,0,0,0): Symbol('l',letter=True),
         (1,0,1,1,0,0): Symbol('m',letter=True),
         (1,0,1,1,1,0): Symbol('n',letter=True),
         (1,0,1,0,1,0): Symbol('o',letter=True),
         (1,1,1,1,0,0): Symbol('p',letter=True),
         (1,1,1,1,1,0): Symbol('q',letter=True),
         (1,1,1,0,1,0): Symbol('r',letter=True),
         (0,1,1,1,0,0): Symbol('s',letter=True),
         (0,1,1,1,1,0): Symbol('t',letter=True),
         (1,0,1,0,0,1): Symbol('u',letter=True),
         (1,1,1,0,0,1): Symbol('v',letter=True),
         (0,1,0,1,1,1): Symbol('w',letter=True),
         (1,0,1,1,0,1): Symbol('x',letter=True),
         (1,0,1,1,1,1): Symbol('y',letter=True),
         (1,0,1,0,1,1): Symbol('z',letter=True),
         (0,0,1,1,1,1): Symbol('#',special=True),
    }

    number_characters = {
        'a': 1,
        'b': 2,
        'c': 3,
        'd': 4,
        'e': 5,
        'f': 6,
        'g': 7,
        'h': 8,
        'i': 9,
        'j': 0,
    }
    
    def __init__(self):
        self.result = ''
        self.shift_on = False
        self.prev_end = None
        self.number = False
        return;

    def import_class_file(self, class_path='helpers/files/class_labels.json'):
        with open(class_path) as json_file:
            return json.load(json_file)
    
    def get_class(self, prediction, class_labels):
        for key, value in class_labels.items():
            if prediction == value:
                return key
    
    def open_preprocess_image(self, image_path):
        img = load_img(image_path, target_size=(150,100))
        processed_img = img_to_array(img)
        processed_img = processed_img / 255.0
        processed_img = processed_img.reshape(1, 150, 100, 3)

        return processed_img

    # @tf.function
    def push_and_predict(self, 
                         image, 
                         character, 
                         classify=True,
                         model_path="helpers/files/cnn_v1-0.h5"):

        if not character.is_valid():
            return;
        box = character.get_bounding_box()
        dots = character.get_dot_coordinates()
        diameter = character.get_dot_diameter()
        end,start,width,combination = get_combination(box, dots, diameter)

        # if combination not in self.symbol_table:
        #     self.result += "*"
        #     return;

        if self.prev_end is not None:
            dist = get_distance(self.prev_end, start)
            if dist > (width**2.3):
                self.result += " "
        self.prev_end = end

        if classify:
            model = load_model(model_path)
            class_labels = self.import_class_file()
            # Crop, process, and predict character
            left, right, top, bottom = box
            char = image.original[top:bottom, left:right]
            image_path = "static/serving/char.png"
            cv2.imwrite(image_path, char)

            processed_img = self.open_preprocess_image(image_path)
            prediction = model.predict(processed_img, verbose=0)
            prediction = prediction.argmax(axis=-1)[0]
            prediction = self.get_class(prediction, class_labels)
            if prediction == 'titik':
                prediction = '.'

            self.result += prediction

        # symbol = self.symbol_table[combination]
        # if symbol.letter() and self.number:
        #     self.number = False
        #     self.result += translate_to_number(symbol.value)
        # elif symbol.letter():
        #     if self.shift_on:
        #         self.result += symbol.value.upper()
        #     else:
        #         self.result += symbol.value
        # else:
        #     if symbol.value == '#':
        #         self.number = True
        return;

    def push(self, character):      
        if not character.is_valid():
            return;
        box = character.get_bounding_box()
        dots = character.get_dot_coordinates()
        diameter = character.get_dot_diameter()
        end,start,width,combination = get_combination(box, dots, diameter)

        if combination not in self.symbol_table:
            self.result += "*"
            return;

        if self.prev_end is not None:
            dist = get_distance(self.prev_end, start)
            if dist > (width**2):
                self.result += " "
        self.prev_end = end

        symbol = self.symbol_table[combination]
        if symbol.letter() and self.number:
            self.number = False
            self.result += translate_to_number(symbol.value)
        elif symbol.letter():
            if self.shift_on:
                self.result += symbol.value.upper()
            else:
                self.result += symbol.value
        else:
            if symbol.value == '#':
                self.number = True
        return;

    def convert_symbols(self):
        temp_result = list(self.result)
        for i, char in enumerate(temp_result):

            # Number
            if char == '#':
                try:
                    temp_result[i+1] = self.number_characters[temp_result[i+1]]
                    del temp_result[i]
                except:
                    pass
            
            temp_result[i] = str(temp_result[i])

        temp_result = ''.join(temp_result)
        self.result = temp_result
        return;

    def digest(self):
        self.convert_symbols()
        return self.result

    def clear(self):
        self.result = ''
        self.shift_on = False
        self.prev_end = None
        self.number = False
        return;
