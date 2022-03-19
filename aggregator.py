import os
import sys
import pandas as pd

cols = ["id", "true.spec.r", "true.spec.g", "true.spec.b", "true.spec.act_lum",
        "true.spec.est_lum", "true.diff.r", "true.diff.g", "true.diff.b",
        "true.diff.act_lum", "true.diff.est_lum", "false.spec.r", "false.spec.g", "false.spec.b",
        "false.spec.act_lum", "false.spec.est_lum", "false.diff.r", "false.diff.g", "false.diff.b",
        "false.diff.act_lum", "false.diff.est_lum"]

OUTPUT = sys.argv[1]

data = []
os.chdir(f"{OUTPUT}data")
print(os.getcwd())
for i in os.listdir():
    csv = pd.DataFrame(data=pd.read_csv(i))
    data.append(csv.values[0])

df = pd.DataFrame(columns=cols, data=data)
df.to_csv("../results.csv", index=False)
