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

def get_results(dir, dir_name):
    count = -1
    count1 = -1
    count2 = -1
    count3 = -1
    with open(dir + dir_name, "r") as f:
        train_ind = []
        test_ind = []
        train_acc = []
        train_f1 = []
        train_fpr = []
        train_fnr = []
        test_acc = []
        test_f1 = []
        test_fpr = []
        test_fnr = []

        train_fpr_all = []
        train_fpr_mean = []
        train_fpr_diff = []
        train_fnr_all = []
        train_fnr_mean = []
        train_fnr_diff = []
        test_fpr_all = []
        test_fpr_mean = []
        test_fpr_diff = []
        test_fnr_all = []
        test_fnr_mean = []
        test_fnr_diff = []
        for line in f:
            if line == "training dataset\n":
                count = 2
                continue
            if count > 0:
                if count == 1:
                    try:
                        match = re.search(r':(\d+\.\d+),\s', line)
                        print("-->", line)
                        num = float(match.group(1))
                        train_ind.append(f"{num:.4f}")
                    except:
                        print("no match")
                count -= 1

            if line == "testing dataset\n":
                count1 = 2
                continue
            if count1 > 0:
                if count1 == 1:
                    try:
                        match = re.search(r':(\d+\.\d+),\s', line)
                        num = float(match.group(1))
                        test_ind.append(f"{num:.4f}")
                    except:
                        print("no match")
                count1 -= 1
            if "overall baseline on training dataset" in line:
                # print("CONTAIN")
                count2 = 4
                continue
            if count2 > 0:
                if count2 == 4:
                    match = re.search(r' (\d+\.\d+)', line)
                    num = float(match.group(1))
                    train_acc.append(f"{num:.4f}")
                if count2 == 3:
                    match = re.search(r' (\d+\.\d+)', line)
                    num = float(match.group(1))
                    train_f1.append(f"{num:.4f}")
                if count2 == 2:
                    match = re.search(r' (\d+\.\d+)', line)
                    num = float(match.group(1))
                    train_fpr.append(f"{num:.4f}")
                if count2 == 1:
                    match = re.search(r' (\d+\.\d+)', line)
                    num = float(match.group(1))
                    train_fnr.append(f"{num:.4f}")
                count2 -= 1

            if "overall baseline on testing dataset" in line:
                count3 = 4
                continue
            if count3 > 0:
                if count3 == 4:
                    match = re.search(r'ACC (\d+\.\d+)', line)
                    num = float(match.group(1))
                    test_acc.append(f"{num:.4f}")
                if count3 == 3:
                    match = re.search(r'F1 (\d+\.\d+)', line)
                    num = float(match.group(1))
                    test_f1.append(f"{num:.4f}")
                if count3 == 2:
                    match = re.search(r' (\d+\.\d+)', line)
                    num = float(match.group(1))
                    test_fpr.append(f"{num:.4f}")
                if count3 == 1:
                    match = re.search(r' (\d+\.\d+)', line)
                    num = float(match.group(1))
                    test_fnr.append(f"{num:.4f}")
                count3 -= 1
            if "-->FPR:[" in line:
                if len(train_fpr_all) == len(test_fpr_all):
                    train_fpr_all.append(line.split(":")[1])

                    pattern = r"mean FPR:(\d+\.\d+)"
                    match = re.search(pattern, line)
                    if match:
                        train_fpr_mean.append(match.group(1))
                    else:
                        print("Value not found")

                    pattern = r"mean FPR diff:(\d+\.\d+)"
                    match = re.search(pattern, line)
                    if match:
                        train_fpr_diff.append(match.group(1))
                    else:
                        print("Value not found")
                else:
                    test_fpr_all.append(line.split(":")[1])

                    pattern = r"mean FPR:(\d+\.\d+)"
                    match = re.search(pattern, line)
                    if match:
                        test_fpr_mean.append(match.group(1))
                    else:
                        print("Value not found")

                    pattern = r"mean FPR diff:(\d+\.\d+)"
                    match = re.search(pattern, line)
                    if match:
                        test_fpr_diff.append(match.group(1))
                    else:
                        print("Value not found")

            if "-->FNR:[" in line:
                if len(train_fnr_all) == len(test_fnr_all):
                    train_fnr_all.append(line.split(":")[1])

                    pattern = r"mean FNR:(\d+\.\d+)"
                    match = re.search(pattern, line)
                    if match:
                        train_fnr_mean.append(match.group(1))
                    else:
                        print("Value not found")

                    pattern = r"mean FNR diff:(\d+\.\d+)"
                    match = re.search(pattern, line)
                    if match:
                        train_fnr_diff.append(match.group(1))
                    else:
                        print("Value not found")
                else:
                    test_fnr_all.append(line.split(":")[1])

                    pattern = r"mean FNR:(\d+\.\d+)"
                    match = re.search(pattern, line)
                    if match:
                        test_fnr_mean.append(match.group(1))
                    else:
                        print("Value not found")

                    pattern = r"mean FNR diff:(\d+\.\d+)"
                    match = re.search(pattern, line)
                    if match:
                        test_fnr_diff.append(match.group(1))
                    else:
                        print("Value not found")

    return train_ind, train_acc, train_f1, train_fpr, train_fnr, train_fpr_all, train_fpr_mean, train_fpr_diff, train_fnr_all, train_fnr_mean, train_fnr_diff,\
           test_ind, test_acc, test_f1, test_fpr, test_fnr, test_fpr_all, test_fpr_mean, test_fpr_diff, test_fnr_all, test_fnr_mean, test_fnr_diff


dir_name = "individual_0.05_0.001_1/figure/result.log"
dir_name0 = "individual_0.1_0.001_1/figure/result.log"
dir_name1 = "individual_0.15_0.001_1/figure/result.log"
dir_name2 = "individual_0.15_0.005_1/figure/result.log"
dir_name3 = "individual_0.12_0.001_1/figure/result.log"
dir_name4 = "individual_0.15_0.01_1/figure/result.log"


dir1 = "./hate_speech_white1/"
dir2 = "./hate_speech_white2/"
dir3 = "./hate_speech_white/"
dir4 = "./hate_speech_white3/"

dir = "./"

train_ind, train_acc, train_f1, train_fpr, train_fnr, train_fpr_all, train_fpr_mean, train_fpr_diff, train_fnr_all, train_fnr_mean, train_fnr_diff,\
test_ind, test_acc, test_f1, test_fpr, test_fnr, test_fpr_all, test_fpr_mean, test_fpr_diff, test_fnr_all, test_fnr_mean, test_fnr_diff = \
    get_results(dir, "result_all.log")
# train_fpr_all = train_fpr_all[:30]
# train_fpr_mean = train_fpr_mean[:30]
# train_fpr_diff = train_fpr_diff[:30]
# train_fnr_all = train_fnr_all[:30]
# train_fnr_mean = train_fnr_mean[:30]
# train_fnr_diff = train_fnr_diff[:30]
# test_fpr_all = train_fpr_all[:30]
# test_fpr_mean = train_fpr_mean[:30]
# test_fpr_diff = train_fpr_diff[:30]
# test_fnr_all = train_fnr_all[:30]
# test_fnr_mean = train_fnr_mean[:30]
# test_fnr_diff = train_fnr_diff[:30]
# train_ind = train_ind[:30]
# train_acc = train_acc[:30]
# train_f1 = train_f1[:30]
# train_fpr = train_fpr[:30]
# train_fnr = train_fnr[:30]
# test_ind = test_ind[:30]
# test_acc = test_acc[:30]
# test_f1 = test_f1[:30]
# test_fpr = test_fpr[:30]
# test_fnr = test_fnr[:30]
datas = {"train_ind": train_ind, "train_acc": train_acc,  "train_f1": train_f1, "train_fpr": train_fpr, "train_fnr": train_fnr,
          "train_fpr_all": train_fpr_all, "train_fpr_mean": train_fpr_mean, "train_fpr_diff": train_fpr_diff,
         "train_fnr_all": train_fnr_all, "train_fnr_mean": train_fnr_mean, "train_fnr_diff": train_fnr_diff,
         "test_ind": test_ind, "test_acc": test_acc, "test_f1": test_f1, "test_fpr": test_fpr, "test_fnr": test_fnr,
         "test_fpr_all": test_fpr_all, "test_fpr_mean": test_fpr_mean, "test_fpr_diff": test_fpr_diff,
         "test_fnr_all": test_fnr_all, "test_fnr_mean": test_fnr_mean, "test_fnr_diff": test_fnr_diff
         }

print(len(train_ind), len(train_acc), len(train_f1), len(train_fpr), len(train_fnr),
      len(test_ind), len(test_acc), len(test_f1), len(test_fpr), len(test_fnr))
print(len(train_fpr_all), len(train_fpr_mean), len(train_fpr_diff), len(train_fnr_all), len(train_fnr_mean), len(train_fnr_diff))
print(len(test_fpr_all), len(test_fpr_mean), len(test_fpr_diff), len(test_fnr_all), len(test_fnr_mean), len(test_fnr_diff))
dataset = pd.DataFrame(datas)

# dataset = pd.concat([dataset1, dataset2, dataset3, dataset4])
print(dataset)
dataset.to_csv("result.csv")





