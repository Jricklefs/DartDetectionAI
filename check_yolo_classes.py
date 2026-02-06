import cv2
import numpy as np
from ultralytics import YOLO

img = cv2.imread(r'C:\Users\clawd\DartGameSystem\DartGameAPI\wwwroot\images\calibrations\cam0_calibration_20260206161352.jpg')
print(f'Image loaded: {img is not None}')
if img is None:
    exit()

h, w = img.shape[:2]
print(f'Image size: {w}x{h}')
center_x, center_y = w/2, h/2

model = YOLO(r'models\dartboard1280imgz_int8_openvino_model', task='detect')
results = model(img, imgsz=1280, conf=0.5, verbose=False)

class_points = {3: [], 4: [], 5: [], 6: []}

for result in results:
    if result.boxes is None:
        continue
    for i in range(len(result.boxes)):
        cls_id = int(result.boxes.cls[i])
        if cls_id in class_points:
            x1, y1, x2, y2 = result.boxes.xyxy[i].cpu().numpy()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            class_points[cls_id].append(dist)

print('Average distance from center by class:')
class_names = {3: 'cal', 4: 'cal1', 5: 'cal2', 6: 'cal3'}
for cls_id in [3, 4, 5, 6]:
    dists = class_points[cls_id]
    if dists:
        avg = np.mean(dists)
        print(f'  Class {cls_id} ({class_names[cls_id]}): avg={avg:.1f}px, count={len(dists)}')
    else:
        print(f'  Class {cls_id} ({class_names[cls_id]}): no detections')
