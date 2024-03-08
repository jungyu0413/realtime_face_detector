import cv2
import numpy as np
import torch
from torchvision import transforms
from src.face_crop import crop


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    idx = True
    while idx:
        idx, image = cap.read()
        image = cv2.flip(image, 1)
        img_height, img_width = image.shape[:2]
        output_image, check = crop(image, preprocess, 224, True, 'cuda')
        if check:
            # model(output_image)
            pass
        else:
            output_image=image
        if idx == False:
            cap.release()
        else:
            cv2.imshow("Output", image)
            k = cv2.waitKey(2) & 0xFF
            if k == 27: # ESC key
                break