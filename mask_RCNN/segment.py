import numpy as np
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = 'mask_RCNN/sam_vit_h_4b8939.pth'
model_type = 'vit_h'

device = 'cpu'

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)

predictor = SamPredictor(sam)

def get_mask(image,boxes):
    predictor.set_image(image)
    masks=[]
    for box in boxes:
        mask,_,_ = predictor.predict(
            box=box,
            multimask_output=False

        )
        mask = process_mask(mask)
        masks.append(mask)

    return masks

def process_mask(mask):

    mask = mask.transpose(1,2,0)
    mask=mask*(np.array([25/255,144/255,255/255,0.6]).reshape(1,1,-1))

    return mask
    