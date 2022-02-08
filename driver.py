import sys
from python.SkinDetector import SkinDetector

s = SkinDetector()

INPUT = sys.argv[1]
OUTPUT = sys.argv[2]

id = INPUT[INPUT.index("/")+1:INPUT.rindex(".")]
print(id)
# s.generate_csv(INPUT, OUTPUT)
# s.process(id, f"./images/{id}.jpg", {"display_points":True, "use_stdevs":True})