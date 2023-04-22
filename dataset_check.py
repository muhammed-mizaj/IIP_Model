import cv2
import sys
# Read the image

path = sys.argv[1]
image = cv2.imread(path)
height, width, _ = image.shape
print("image size {} {}".format(height, width))

# Open the file with YOLO coordinates
with open(path.replace("images", "labels").replace(".jpg", ".txt"), "r") as file:
    for line in file:
        # Split the line to get the class label, confidence, and box coordinates
        class_label, x_norm, y_norm, w_norm, h_norm = line.strip().split()
        print(class_label)
        print(x_norm, y_norm, w_norm, h_norm)
        # Convert the coordinates from strings to integers
        x_norm, y_norm, w_norm, h_norm = float(x_norm), float(
            y_norm), float(w_norm), float(h_norm)

        # Convert the normalized coordinates to pixel coordinates
        x, y = int(x_norm * width), int(y_norm * height)
        w, h = int(w_norm * width), int(h_norm * height)

        # Calculate the top-left and bottom-right coordinates of the bounding box
        x1, y1 = x - w//2, y - h//2
        x2, y2 = x + w//2, y + h//2

        print(x1, y1, x2, y2)

        print("\n\n")

        # Draw the bounding box on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        text = f"{class_label}"
        cv2.putText(image, text, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

image = cv2.resize(image, (1000, 1000))
# Show the image with the bounding boxes
cv2.imshow("Image with YOLO bounding boxes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
