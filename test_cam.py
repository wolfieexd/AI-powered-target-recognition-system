import cv2
import time

print("Testing camera 0...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Failed to open camera 0 with DSHOW")
    cap = cv2.VideoCapture(0)
    
if not cap.isOpened():
    print("Failed to open camera 0 entirely")
else:
    print("Camera opened successfully.")
    for i in range(5):
        ret, frame = cap.read()
        print(f"Frame {i}: ret={ret}, shape={frame.shape if frame is not None else None}")
        time.sleep(0.5)
        
cap.release()
print("Done.")
