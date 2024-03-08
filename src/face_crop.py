import cv2
from src.faceboxes_detector import *
import src.faceboxes_detector as faceboxes_detector
from PIL import Image

def crop(image, preprocess, input_size, use_gpu, device):
    detector = faceboxes_detector.FaceBoxesDetector('FaceBoxes', '/workspace/realtime_face_detector/src/weights/FaceBoxesV2.pth', use_gpu, device)

    det_box_scale = 1.2

    image_height, image_width, _ = image.shape
    detections, check = detector.detect(image, 600, 0.8, 'max', 1)
    
    if check:
            # Get bounding box coords. 
            det_xmin = detections[0][2]
            det_ymin = detections[0][3]
            det_width = detections[0][4]
            det_height = detections[0][5]
            det_xmax = det_xmin + det_width - 1
            det_ymax = det_ymin + det_height - 1

            det_xmin -= int(det_width * (det_box_scale-1)/2)
            # remove a part of top area for alignment, see paper for details
            det_ymin += int(det_height * (det_box_scale-1)/2)
            det_xmax += int(det_width * (det_box_scale-1)/2)
            det_ymax += int(det_height * (det_box_scale-1)/2)
            det_xmin = max(det_xmin, 0)
            det_ymin = max(det_ymin, 0)
            det_xmax = min(det_xmax, image_width-1)
            det_ymax = min(det_ymax, image_height-1)
            det_width = det_xmax - det_xmin + 1
            det_height = det_ymax - det_ymin + 1
            
            ### Crop by bounding box. 
            # Comparing width & height, we adjust the bbox to square-crop (to conserve aspect ratio).
            if det_width > det_height: 
                buffer = int((det_width - det_height)/2)
                det_ymin -= buffer
                det_ymax += buffer
            
            elif det_width < det_height:
                buffer = int((det_height - det_width)/2)
                det_xmin -= buffer
                det_xmax += buffer
            
            
            #cv2.rectangle(image, (det_xmin, det_ymin), (det_xmax, det_ymax), (0, 0, 255), 2)
            det_crop = image[det_ymin:det_ymax, det_xmin:det_xmax, :]
            det_crop = cv2.resize(det_crop, (input_size, input_size))
            
            inputs = Image.fromarray(det_crop.astype('uint8'), 'RGB')
            inputs = preprocess(inputs).to(device).unsqueeze(0)
            #inputs = inputs.reshape(input_size, input_size, 3)
            
            
            print(f'face box : {check}')
            return inputs, check
        
    else:
        print(f'face box : {check}')
        return check, check
        
