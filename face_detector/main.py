import cv2
from facenet_pytorch import MTCNN
import torch
from PIL import Image

# Load the face detector
mtcnn = MTCNN(keep_all=True, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Load the image using OpenCV
image = cv2.imread("F:\downloads\IMG_20211103_093422.jpg")

# Convert the image to RGB format
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the image to a PIL Image (required by MTCNN)
image_pil = Image.fromarray(image_rgb)

# Detect faces
boxes, _ = mtcnn.detect(image_pil)

# Draw bounding boxes around detected faces
if boxes is not None:
    for box in boxes:
        cv2.rectangle(image, 
                      (int(box[0]), int(box[1])), 
                      (int(box[2]), int(box[3])), 
                      (0, 255, 0), 2)

# Show the image with bounding boxes
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
