import cv2
from PIL import Image
import pytesseract
import numpy as np


def get_rows(img_path):
    image = cv2.imread(img_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 0, 150)
    # cv2.imshow("edges",edges)

    # Perform line detection using the Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=0, maxLineGap=20)
    # cv2.imshow("lines",lines)

    # Sort the lines by their y-coordinate
    lines = sorted(lines, key=lambda line: line[0][1])

    # Get the y-coordinates of the lines
    y_coordinates = [line[0][1] for line in lines]


    new_y_cordinates=[]
    for i in y_coordinates:
        if(len(new_y_cordinates)==0):
            new_y_cordinates.append(i)
        else:
            if(abs(new_y_cordinates[-1]-i)>10):
                new_y_cordinates.append(i)

    # print(new_y_cordinates)
    results=[]
    for i in range(len(new_y_cordinates)-1):
        y1 = new_y_cordinates[i]
        y2 = new_y_cordinates[i+1]
        image_part = image[y1:y2+20, :]
        # cv2.imshow("image",image_part)
        # cv2.waitKey(0)
        results.append(pytesseract.image_to_string(Image.fromarray(image_part)))
    print(results)
        