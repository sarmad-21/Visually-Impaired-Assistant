import cv2
import time

def capture_image(filename="object.jpg"):
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    time.sleep(3)
    result, image = cam.read() #read frame from camera (result is boolean comfirming image captured)
    if result:
        cv2.imwrite(filename, image)
        print('Image captured and saved')
    else:
         print("Failed to save/capture image")
    cam.release()
