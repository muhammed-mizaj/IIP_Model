import pytesseract
import cv2
import numpy as np
from sklearn.cluster import KMeans
# Load the image
img = cv2.imread('unit_1.jpg')
pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'
# Apply OCR using Pytesseract
data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)


colors = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
    (255, 165, 0),  # Orange
    (128, 0, 128),  # Purple
    (255, 192, 203),  # Pink
    (165, 42, 42),  # Brown
    (0, 128, 128),  # Teal
    (0, 0, 128),  # Navy
    (128, 128, 0),  # Olive
    (128, 128, 128),
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
    (255, 165, 0),  # Orange
    (128, 0, 128),  # Purple
    (255, 192, 203),  # Pink
    (165, 42, 42),  # Brown
    (0, 128, 128),  # Teal
    (0, 0, 128),  # Navy
    (128, 128, 0),  # Olive
    (128, 128, 128)  # Gray
]

y_tolerance = 20
n_boxes = len(data['text'])
line_boxes = [[]]
words = [[]]
line_box_index = 0
current_y = -1
for i in range(n_boxes):
    if int(data['conf'][i]) > 60:
        (x, y, w, h) = (data['left'][i], data['top']
                        [i], data['width'][i], data['height'][i])
        if current_y == -1:
            current_y = y
            line_boxes[line_box_index].append((x, y, w, h))
            words[line_box_index].append(data['text'][i])
        else:
            if(abs(current_y - y) <= y_tolerance):
                line_boxes[line_box_index].append((x, y, w, h))
                words[line_box_index].append(data['text'][i])
            else:
                line_box_index = line_box_index+1
                line_boxes.append([])
                line_boxes[line_box_index].append((x, y, w, h))
                words.append([])
                words[line_box_index].append(data['text'][i])
            current_y = y


line_boxes_new = []
center_y_coordinates_for_clustering = []
word_lines = []
for index, i in enumerate(line_boxes):
    # print(index)
    x1, y1 = i[0][0], i[0][1]
    x2, y2 = i[-1][0]+i[-1][2], i[-1][1]+i[-1][3]
    line_boxes_new.append((x1, y1, x2, y2))
    center_y_coordinates_for_clustering.append((y2+y1)/2)
    word_lines.append(" ".join(words[index]))
    # cv2.rectangle(img,(x1,y1),(x2,y2),colors[index],2)


center_y_coordinates_for_clustering = np.array(
    center_y_coordinates_for_clustering)

num_clusters = 5


kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit([[x] for x in center_y_coordinates_for_clustering])


# print(word_lines)
final_items = ["" for i in range(num_clusters)]

# print(kmeans.labels_)
for i in range(len(line_boxes_new)):
    label = kmeans.labels_[i]
    final_items[label] = final_items[label]+word_lines[i]
    color = colors[label]
    x1, y1, x2, y2 = line_boxes_new[i]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

print("The Items are ")
for index, i in enumerate(final_items):
    print(str(index)+"."+i)


# Display image with bounding boxes
img = cv2.resize(img, (700, 700))
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
