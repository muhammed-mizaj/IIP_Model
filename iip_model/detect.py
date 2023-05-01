from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


config_path = 'config/yolov3.cfg'
weights_path = sys.argv[1]
class_path = 'config/coco.names'
img_size = 416
conf_thres = 0.000
nms_thres = 0.005
# Load model and weights
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.cuda()
model.eval()
classes = load_classes(class_path)
cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def detect_image(img):
    # scale and pad image
    img_arr = np.array(img)

    if (len(img_arr.shape) < 3):
        img_arr = np.stack((img,) * 3, axis=-1)
    img = Image.fromarray(img_arr)

    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([transforms.Resize((imh, imw)),
                                         transforms.Pad((max(int((imh-imw)/2), 0),
                                                         max(int(
                                                             (imw-imh)/2), 0), max(int((imh-imw)/2), 0),
                                                         max(int((imw-imh)/2), 0)), (128, 128, 128)),
                                         transforms.ToTensor(),
                                         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        # for i in detections:
        #     print(i)

        detections = non_max_suppression(Tensor.cpu(detections), 80,
                                         conf_thres, nms_thres)

    return detections[0]


# load image and get detections
img_path = sys.argv[2]
prev_time = time.time()
img = Image.open(img_path)
detections = detect_image(img)
# print(detections)
inference_time = datetime.timedelta(seconds=time.time() - prev_time)
print('Inference Time: %s' % (inference_time))
# Get bounding-box colors
cmap = plt.get_cmap('plasma')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]
img = np.array(img)
plt.figure()
fig, ax = plt.subplots(1, figsize=(12, 9))
ax.imshow(img)
pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
unpad_h = img_size - pad_y
unpad_w = img_size - pad_x
if detections is not None:
    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    bbox_colors = random.sample(colors, n_cls_preds)
    # browse detections and draw bounding boxes
    new_detections = []

    for i in range(n_cls_preds):
        detect_arr = list(filter(lambda a: a[6] == i, detections))
        if (len(detect_arr) > 0):
            new_detections.append(detect_arr[0])

    print(len(new_detections))
    # for i in new_detections:
    #     print(i)
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in new_detections:
     #    print(int(cls_pred))
     #    print(classes)
        # print("{} {} {} {} {} {} {}".format(
        #     x1, y1, x2, y2, conf, cls_conf, cls_pred))
        box_h = ((y2 - y1) / unpad_h) * img.shape[0]
        box_w = ((x2 - x1) / unpad_w) * img.shape[1]
        y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
        x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
        color = bbox_colors[int(np.where(
            unique_labels == int(cls_pred))[0])]
        bbox = patches.Rectangle((x1, y1), box_w, box_h,
                                 linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(bbox)
        plt.text(x1, y1, s=classes[int(cls_pred)],
                 color='white', verticalalignment='top',
                 bbox={'color': color, 'pad': 0})

plt.axis('off')
# save image
plt.savefig(img_path.replace("images", "detections").replace(".jpg", "-det.jpg"),
            bbox_inches='tight', pad_inches=0.0)
plt.show()
