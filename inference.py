import sys

sys.path.append("../inputs")

import ubiquant

env = ubiquant.make_env()
iter_test = env.iter_test()

for n, (test_df, sample_prediction_df) in enumerate(iter_test):
    sample_prediction_df["target"] = 0.0
    env.predict(sample_prediction_df)
