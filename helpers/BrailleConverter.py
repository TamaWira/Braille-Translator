import cv2
import imutils
import numpy as np

from imutils import contours
from imutils import perspective

def read_and_resize_img(path, scale):

    img = cv2.imread(path)
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    dim = (width, height)
    
    # resize image
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    return img, resized_img

def preprocess_img(img):

    gray = cv2.GaussianBlur(img, (7, 7), 0)

    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 10)
    
    return thresh

def detect_edge(img):
    
    edged = cv2.Canny(img, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    
    return edged

def find_contour(thresh):
    # Find Contour
    cnts = cv2.findContours(thresh, 
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    (cnts, _) = contours.sort_contours(cnts)

    return cnts

def draw_contour(cnts, img):
    cnts_area = []
    contour_img = img.copy()
    blank_img = np.zeros(img.shape, np.uint8)

    # loop over the contours
    for c in cnts:
        area = cv2.contourArea(c)
        cnts_area.append(area)

        # compute the center of the contour
        M = cv2.moments(c)
        if M["m00"] <= 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        cv2.circle(blank_img, (cX, cY), 4, (255, 255, 255), -1)
        # blank_img = cv2.bitwise_not(blank_img)

        # draw the contour in rectangle shape
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(contour_img,(x,y),(x+w,y+h),(0,255,0), 1)
    
    return cv2.bitwise_not(cv2.blur(blank_img, (3,3))), contour_img, cnts_area