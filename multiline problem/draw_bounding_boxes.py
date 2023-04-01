from PIL import Image
import pytesseract
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'


boxes=pytesseract.image_to_boxes('test.jpg',output_type=pytesseract.Output.DICT)
img = cv2.imread('test.jpg')


box_array=[]
n_boxes = len(boxes['page'])
for i in range(n_boxes):
    (x1, y1, x2, y2) = (boxes['left'][i], boxes['top'][i], boxes['right'][i], boxes['bottom'][i])
    box_array.append([x1,y1,x2,y2])
box_array=np.array(box_array)


# num_clusters = 14

# distances=pairwise_distances(box_array,metric='cosine')
# kmeans = KMeans(n_clusters=num_clusters,init='k-means++',algorithm='full')
# kmeans.fit(box_array,sample_weight=distances)


# colors = [
#     (255, 0, 0),  # Red
#     (0, 255, 0),  # Green
#     (0, 0, 255),  # Blue
#     (255, 255, 0),  # Yellow
#     (0, 255, 255),  # Cyan
#     (255, 0, 255),  # Magenta
#     (255, 165, 0),  # Orange
#     (128, 0, 128),  # Purple
#     (255, 192, 203),  # Pink
#     (165, 42, 42),  # Brown
#     (0, 128, 128),  # Teal
#     (0, 0, 128),  # Navy
#     (128, 128, 0),  # Olive
#     (128, 128, 128)  # Gray
# ]

# print(kmeans.labels_)
# for i in range(len(box_array)):
#     label = kmeans.labels_[i]  
#     color = colors[label] 
#     x1, y1, x2, y2 = box_array[i]  
#     cv2.rectangle(img, (x1, img.shape[0]-y1), (x2, img.shape[0]-y2), color, 2)


# cv2.imshow('Image with clustered bounding boxes', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()