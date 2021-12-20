from SkinDetector import SkinDetector
import json

s = SkinDetector()


id = "morgan_freeman"
# data = s.generate_json("morgan_freeman", "../images/morgan_freeman.jpg")
# print(data)

s.generate_csv("morgan_freeman", "../images/morgan_freeman.jpg")

# f = open(f"../results/data/{id}.json", "w")
# json.dump(data, f)
# f.close()


# s.process("testing",  "../images/ariana_grande.jpg", {"display_points":True})