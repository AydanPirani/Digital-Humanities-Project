import sys
from python.SkinDetector import SkinDetector

s = SkinDetector()
id = sys.argv[1].split(".")[0]
s.generate_csv(id, f"./images/{id}.jpg")
