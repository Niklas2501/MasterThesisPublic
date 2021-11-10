import pandas as pd
import numpy as np

df = pd.read_pickle('../data/datasets/unprocessed_data/9_failure_sm_1m1_pm_06_07_2021/sensor_topics.pkl')

index = df.index.values

diff = index[1:] - index[0:-1]

print(np.mean(diff))