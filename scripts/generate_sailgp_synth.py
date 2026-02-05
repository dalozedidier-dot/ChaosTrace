
import numpy as np, pandas as pd

def generate(path="sample_timeseries.csv"):
    t = np.arange(0,120,0.05)
    speed = 35 + 2*np.sin(t/10) + np.random.normal(0,0.3,len(t))
    foil = 0.8 + 0.05*np.sin(t/5)

    foil[(t>55)&(t<65)] = 0.05
    foil[(t>90)&(t<95)] = 0.1

    pd.DataFrame({"time_s":t,"boat_speed":speed,"foil_height_m":foil}).to_csv(path,index=False)

if __name__ == "__main__":
    generate()
