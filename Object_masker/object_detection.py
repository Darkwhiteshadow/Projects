import torch 
import cv2
from torchvision.transforms import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

detector = fasterrcnn_resnet50_fpn(pretrained=True,progress=False)
detector.eval()

    
def decode_preds(preds,threshold=0.7,target=None):
    coco_classes = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "TV",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]
    preds = preds[0]

    boxes = preds['boxes']
    labels = preds['labels']
    scores = preds['scores']

    results = []

    for box,label,score in zip(boxes,labels,scores):
        if score >=threshold:
            
            if target ==  coco_classes[label.item()].strip('\n'):
                print('entered')
                results.append(box)
                print(results)
            elif target == None:
                results.append(box)
    return results



        
    

def detect(image):

    tensor = transforms.ToTensor()(image)
    tensor = tensor.unsqueeze(0)

    preds = detector(tensor)

    return preds

if __name__ == '__main__':
    path = 'test.jpg'
    preds = detect(path)
    