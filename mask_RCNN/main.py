from object_detection import detect,decode_preds
import cv2
import matplotlib.pyplot as plt
from segment import get_mask
from visualize import visualize_preds


path = r"F:\New folder (2)\1288540.jpg"

image = cv2.imread(path)

preds = detect(image)

predictions = decode_preds(preds,threshold=0.7,target='car')

masks = get_mask(image,predictions[0]['boxes'])   

fig , ax  = plt.subplots(1,2,figsize=(20,10))
ax[0].imshow(mask)
for mask in masks:
    ax.imshow(mask)

plt.show()