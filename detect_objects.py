import cv2
import numpy as np
import sys

# Load YOLOv3 trained weights and config file
weights_path = 'yolov3.weights'
config_path = 'yolov3.cfg'

# Load COCO object classes
classes_path = 'coco.names'
with open(classes_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Set up the network
net = cv2.dnn.readNet(weights_path, config_path)

# Load the input image from argument
if len(sys.argv) < 2:
    print("Usage: python detect_objects.py <image_file>")
    sys.exit(1)

image_path = sys.argv[1]
image = cv2.imread(image_path)
if image is None:
    print(f"Could not open or find the image: {image_path}")
    sys.exit(1)

height, width = image.shape[:2]

# Create a blob from the image
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Get output layer names
layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
except IndexError:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


# Feed blob through the network and collect the detections
layer_outputs = net.forward(output_layers)

boxes = []
confidences = []
class_ids = []

# Iterate over detections and filter by confidence threshold
for output in layer_outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            box = detection[0:4] * np.array([width, height, width, height])
            (centerX, centerY, w, h) = box.astype("int")
            x = int(centerX - (w / 2))
            y = int(centerY - (h / 2))

            boxes.append([x, y, int(w), int(h)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply Non-Maxima Suppression to suppress weak overlapping boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw bounding boxes and labels on the image
if len(indices) > 0:
    # In some OpenCV versions, indices is a 2D array
    if isinstance(indices, (list, np.ndarray)) and len(indices.shape) > 1:
        indices = indices.flatten()
        
    for i in indices:
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]
        confidence = confidences[i]
        color = (0, 255, 0)  # Green box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = f"{label}: {confidence:.2f}"
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Show the output image
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
