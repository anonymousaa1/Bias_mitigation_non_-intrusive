import random

from datasets import load_dataset, load_metric

import pandas as pd
import copy
import torch

from tqdm import tqdm
import numpy as np
import json
import re
from pandas.core.frame import DataFrame

from sklearn.model_selection import train_test_split

seed = 999
random.seed(seed)

fur_all = [0.3095890410958904, 0.30409356725146197, 0.2304147465437788, 0.3333333333333333, 0.3473684210526316, 0.6438356164383562, 0.45555555555555555, 0.2440944881889764]
fur_base = 0.337645537
identities = ["male", "female", "homosexual", "christian", "muslim", "jewish", "black", "white"]
# identities = ["male", "female", "homosexual", "muslim", "black", "white"]
# identities = ["male", "female", "homosexual", "christian", "muslim", "black", "white"]
gender_identities = ["male", "female", "homosexual"]
religion_identities = ["christian", "muslim", "jewish"]
# religion_identities = ["muslim"]
# religion_identities = ["christian", "muslim",]
race_identities = ["black", "white"]

gender_fur_diff = []
religion_fur_diff = []
race_fur_diff = []
all_diff = []
for i in range(0, len(identities)):
    identity = identities[i]
    fur = fur_all[i]
    fur_diff = abs(fur - fur_base)
    # all_diff.append(fur - fur_base)
    all_diff.append("{:.4f}".format(fur - fur_base))

    if identity in gender_identities:
        gender_fur_diff.append(fur_diff)
    elif identity in religion_identities:
        religion_fur_diff.append(fur_diff)
    elif identity in race_identities:
        race_fur_diff.append(fur_diff)

print("gender: {:.4f}".format(sum(gender_fur_diff)/float(3)))
print("religion: {:.4f}".format(sum(religion_fur_diff)/float(3)))
print("race: {:.4f}".format(sum(race_fur_diff)/float(2)))
print(all_diff)
new_all_diff = []
for i in all_diff:
    new_all_diff.append(float(i))
print(new_all_diff)
print(set(new_all_diff[:3]))
print(set(new_all_diff[3:6]))
print(set(new_all_diff[6:8]))
