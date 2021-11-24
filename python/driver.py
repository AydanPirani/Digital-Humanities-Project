from SkinDetector import SkinDetector
import cv2

# img = cv2.imread("../images/morgan_freeman.jpg")

s = SkinDetector()

from SkinDetector import SkinDetector
s = SkinDetector()
# s.process(img, "morgan_freeman")

# data = s.generate_json("morgan_freeman", "../images/morgan_freeman.jpg")
# print(data)
s.process("testing",  "../images/ariana_grande.jpg", {"display_points":True})