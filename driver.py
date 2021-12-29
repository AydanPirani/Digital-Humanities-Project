import sys
from python.SkinDetector import SkinDetector
import json

s = SkinDetector()


id = "morgan_freeman"
#id = sys.argv[1].split(".")[0]

#data = s.generate_json("morgan_freeman", "/opt/project/images/morgan_freeman.jpg")
#print(data)

#s.generate_csv(id, f"images/{id}.jpg")
s.generate_csv(id, f"/opt/project/images/{id}.jpg")

# f = open(f"../results/data/{id}.json", "w")
# json.dump(data, f)
# f.close()


# s.process("testing",  "../images/ariana_grande.jpg", {"display_points":True})