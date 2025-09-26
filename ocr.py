from camera import capture_image
import cv2
import pytesseract


def text_recognition():
    capture_image("text.jpg")
    print("Running Tesseract OCR ...")
    image = cv2.imread("text.jpg")
    cv2.imshow("Captured Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binarized = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  #how threshold calculated (threshold value)
        cv2.THRESH_BINARY,               #how threshold applied (binarization)
        31, 7
    )
    cv2.imshow("Grayscale", gray)
    cv2.imshow("Binarized", binarized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    text = pytesseract.image_to_string(binarized).strip()  #extract text + print remove leading/trailing whitespaces
    print(text)
    return text
