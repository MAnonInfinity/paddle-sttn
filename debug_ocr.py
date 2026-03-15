import cv2
import json

cap = cv2.VideoCapture('videos/9.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

# extract a few frames from middle
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_indices = [total_frames // 4, total_frames // 2, total_frames * 3 // 4]

from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=False, lang='ch', use_gpu=False, det_db_thresh=0.2, det_db_box_thresh=0.4)

results = []
for idx in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(f'frame_{idx}.jpg', frame)
        res = ocr.ocr(frame, cls=False, rec=True)
        results.append({"frame": idx, "ocr": res})

with open('debug_ocr.json', 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Saved frames and ocr results")
