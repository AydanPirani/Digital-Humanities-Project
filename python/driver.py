from SkinDetector import SkinDetector
import cv2

# img = cv2.imread("../images/morgan_freeman.jpg")

s = SkinDetector()
# s.process(img, "morgan_freeman")

from SkinDetector import SkinDetector
s = SkinDetector()
data = s.generate_json("morgan_freeman", "../images/morgan_freeman.jpg")
print(data)