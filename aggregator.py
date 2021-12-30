import pandas as pd
import os


cols = ["id", "true/spec/r", "true/spec/g", "true/spec/b", "true/spec/act_lum",
        "true/spec/esp_lum", "true/diff/r", "true/diff/g", "true/diff/b",
        "true/diff/act_lum", "true/diff/esp_lum", "false/spec/r", "false/spec/g",
        "false/spec/b", "false/spec/act_lum", "false/spec/esp_lum", "false/diff/r",
        "false/diff/g", "false/diff/b", "false/diff/act_lum", "false/diff/esp_lum"]



data = []
os.chdir("./results/data")
for i in os.listdir():
    csv = pd.DataFrame(data=pd.read_csv(i))
    data.append(csv.values[0])

df = pd.DataFrame(columns=cols, data=data)
df.to_csv(f"../results.csv", index=False)

