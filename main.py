import cv2

capture = cv2.VideoCapture(0)

while True:
    success, img = capture.read()
    cv2.imshow("Image", img)
    cv2.waitKey(1)