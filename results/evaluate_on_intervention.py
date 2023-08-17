import numpy as np
import csv
import matplotlib.pyplot as plt

base_group_fairness = [0.04441648156674771, 0.007068208924357444, 0.007455355652967621, 0.054687256543405116,
                       0.031811439549941034, 0.18960789146404, 0.039440807086429275, 0.08905452026846694,
                       0.05609194821358343, 0.03184559955540667, 0.03374838454070328, 0.024950467312944924,
                       0.00828921014535866]
base_FP_diff = [0.0702944437137113, 0.015032510034295121, 0.04335925575234517, 0.04359726598678526, 0.04076941162128511,
                0.25652926152747646, 0.09921810421631916, 0.11311335539970593, 0.09207302810572579, 0.07115725648334026,
                0.1459523384505534, 0.10617961117782612, 0.08816931958262886]

base_FN_diff = [0.046671732172871905, 0.001177730192719488, False, 0.0003005372102633484, 0.0004530925115600673,
                0.09690970150126958, 0.0007503797653690587, 0.002155603140613846, 0.00033742132243202647, 0.0017153646013216388,
                0.004857383872373167, 0.0020963653633231626, 0.0022193968593861545]


def plot_single(aces, metrics, x=None):
    y = aces
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.set_title(metrics)
    plt.title(metrics)
    plt.xlabel('terms')
    plt.ylabel('ACE')
    if x == None:
        x = list(range(0, len(aces)))
    else:
        x = x
        # plt.xticks(fontsize=6)
        plt.xticks(rotation=90, fontsize=8)
    plt.scatter(x, y, c='r', marker='o')
    plt.savefig(metrics + ".png")
    plt.close()



# result_file = "layer_12_result.csv"
#
# csvFile = open(result_file, "r")
# reader = csv.reader(csvFile)
# all_fair_score = []
# for item in reader:
#     # print(item)
#     if "fair_score" in item[0]:
#         score = eval(item[0].split("fair_score")[1])
#         all_fair_score.append(score)

result_file = "layer_1_result.txt"
f = open(result_file, "r")
all_fair_score = eval(f.read())


def get_diff(neuron_result, base_result):
    result = []
    for i in range(0, len(neuron_result)):
        if neuron_result[i] == False:
            continue
        # result.append(abs(neuron_result[i] - base_result[i]))
        result.append(neuron_result[i] - base_result[i])
    return result

group_fair_score = []
FP_diff_score = []
FN_diff_score = []
for neuron_result in all_fair_score:
    # group_fair_score.append(neuron_result[0][0])
    # FP_diff_score.append(neuron_result[1][0])
    # FN_diff_score.append(neuron_result[2][0])

    group_fair_score.append(get_diff(neuron_result[0][0], base_group_fairness))
    FP_diff_score.append(get_diff(neuron_result[1][0], base_FP_diff))
    FN_diff_score.append(get_diff(neuron_result[2][0], base_FN_diff))

# group_fair_score = get_diff(group_fair_score, base_group_fairness)
# FP_diff_score = get_diff(FP_diff_score, base_FP_diff)
# FN_diff_score = get_diff(FN_diff_score, base_FN_diff)

# evaluating group fairness score
ace_group = []
for score in group_fair_score:
    ace_group.append(sum(score))

print("-->ace_group", ace_group)

# evaluating FP difference rate
ace_FP = []
for score in FP_diff_score:
    ace_FP.append(sum(score))


print("-->ace_FP", ace_FP)

# evaluating FN difference rate
ace_FN = []
for score in FN_diff_score:
    ace_FN.append(sum(score))

print("-->ace_FP", ace_FP)

# plot_single(ace_group, "group_2")
# plot_single(ace_FP, "FP_equality_diff_2")
# plot_single(ace_FN, "FN_equality_diff_2")


ace_group_1 = ace_group
ace_FP_1 = ace_FP
ace_FN_1 = ace_FN



result_file = "layer_2_result.txt"
f = open(result_file, "r")
all_fair_score = eval(f.read())


def get_diff(neuron_result, base_result):
    result = []
    for i in range(0, len(neuron_result)):
        if neuron_result[i] == False:
            continue
        # result.append(abs(neuron_result[i] - base_result[i]))
        result.append(neuron_result[i] - base_result[i])
    return result

group_fair_score = []
FP_diff_score = []
FN_diff_score = []
for neuron_result in all_fair_score:
    # group_fair_score.append(neuron_result[0][0])
    # FP_diff_score.append(neuron_result[1][0])
    # FN_diff_score.append(neuron_result[2][0])

    group_fair_score.append(get_diff(neuron_result[0][0], base_group_fairness))
    FP_diff_score.append(get_diff(neuron_result[1][0], base_FP_diff))
    FN_diff_score.append(get_diff(neuron_result[2][0], base_FN_diff))

# group_fair_score = get_diff(group_fair_score, base_group_fairness)
# FP_diff_score = get_diff(FP_diff_score, base_FP_diff)
# FN_diff_score = get_diff(FN_diff_score, base_FN_diff)

# evaluating group fairness score
ace_group = []
for score in group_fair_score:
    ace_group.append(sum(score))

print("-->ace_group", ace_group)

# evaluating FP difference rate
ace_FP = []
for score in FP_diff_score:
    ace_FP.append(sum(score))


print("-->ace_FP", ace_FP)

# evaluating FN difference rate
ace_FN = []
for score in FN_diff_score:
    ace_FN.append(sum(score))

print("-->ace_FP", ace_FP)

ace_group_all = ace_group_1 + ace_group
ace_FP_all = ace_FP_1 + ace_FP
ace_FN_all = ace_FN_1 + ace_FN

plot_single(ace_group_all, "group")
plot_single(ace_FP_all, "FP_equality_diff")
plot_single(ace_FN_all, "FN_equality_diff")

# score = get_diff(FP_diff_score[0], base_group_fairness)
# gender_terms= ["lesbian", "gay", "bisexual", "transgender", "trans", "queer", "lgbt", "lgbtq", "homosexual",
#                "straight", "heterosexual", "male", "female"]
#
# plot_single(score, "group score", gender_terms)
