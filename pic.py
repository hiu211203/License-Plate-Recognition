from PIL import Image
import cv2
import torch
import function.utils_rotate as utils_rotate
import argparse
import function.helper as helper

# Parse the input arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i1', '--image1', required=True, help='path to input image for stage 1')
ap.add_argument('-i2', '--image2', required=True, help='path to input image for stage 2')
args = ap.parse_args()

# Load the YOLO models
yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.60

# Read the input images
img_stage1 = cv2.imread(args.image1)
img_stage2 = cv2.imread(args.image2)

# Detect license plates in the first image (Stage 1)
plates = yolo_LP_detect(img_stage1, size=640)
list_plates = plates.pandas().xyxy[0].values.tolist()

if len(list_plates) == 0:
    print("No license plates detected in Stage 1 image.")
else:
    for plate in list_plates:
        x = int(plate[0])  # xmin
        y = int(plate[1])  # ymin
        w = int(plate[2] - plate[0])  # xmax - xmin
        h = int(plate[3] - plate[1])  # ymax - ymin  
        cv2.rectangle(img_stage1, (x, y), (x + w, y + h), color=(0, 0, 225), thickness=2)

# Recognize characters in the second image (Stage 2)
lp = helper.read_plate(yolo_license_plate, img_stage2)
if lp != "unknown":
    # Adjust text position and size to fit within the image
    font_scale = max(0.4, img_stage2.shape[1] / 1000)
    text_size, _ = cv2.getTextSize(lp, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    text_x = 10
    text_y = min(img_stage2.shape[0] - 10, text_size[1] + 10)
    cv2.putText(img_stage2, lp, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (36, 255, 12), 2)
else:
    print("No characters recognized in Stage 2 image.")

# Save and display the images
cv2.imwrite('stage1_detected.jpg', img_stage1)
cv2.imwrite('stage2_recognized.jpg', img_stage2)

# Display the images
cv2.imshow('Stage 1: License Plate Detection', img_stage1)
cv2.imshow('Stage 2: Character Detection', img_stage2)
cv2.waitKey()
cv2.destroyAllWindows()
