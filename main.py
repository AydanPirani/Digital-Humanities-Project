import cv2
from FaceTools import FaceDetectionTools

capture = cv2.VideoCapture(0)
FDT = FaceDetectionTools()

while True:
    success, img = capture.read()
    FDT.detect_face(img)
    FDT.generate_mesh(img)
    FDT.draw(img)
    cv2.imshow("Image", img)
    cv2.waitKey(1)