#!/usr/bin/env python3


import torch

import numpy as np

import glob

import os
import sys
import scipy.stats
import tabulate


def analyze_results_given_dir_pattern(dir_pattern):
    results = []
    for dirname in glob.glob(dir_pattern):
        loaded = torch.load(os.path.join(dirname, 'model_best.pth.tar'), map_location='cpu')
        best_prec1 = loaded['best_acc1']
        try:
            weight = loaded['state_dict']['module.fc.weight'].numpy()
        except KeyError:
            keys = loaded['state_dict'].keys()
            if 'module.linear.weight' in keys:
                weight = loaded['state_dict']['module.linear.weight'].numpy()
            elif 'module.classifier.weight' in keys:  # Conv 1x1
                weight = loaded['state_dict']['module.classifier.weight'].numpy()
            else:
                raise KeyError("Don't know what key to use, available keys are", keys())
        real_sparsity = 1 - np.count_nonzero(weight)/weight.size
        results.append([real_sparsity, best_prec1])

    results = sorted(results, key=lambda pair: pair[0])
    results = np.asarray(results)
    return results


def get_basic_statistics(results_dict):
    out = []
    for key in results_dict:
        try:
            sparsity = results_dict[key][0, 0]
            mean_acc = np.mean(results_dict[key][:, 1])
            stdev = np.std(results_dict[key][:, 1])
            line = [key, sparsity, mean_acc, stdev, len(results_dict[key])]
            out.append(line)
        except IndexError:
            pass
    return out


targets = sys.argv[2:]
prefix = len(os.path.commonprefix(targets))
results = {}

for target in targets:
    path_pattern = os.path.realpath(target)
    path_pattern += '/save_run_*'
    target_name = target[prefix-3:]
    results[target_name] = analyze_results_given_dir_pattern(path_pattern)

# brief
if 'b' in sys.argv[1]:
    table_output = tabulate.tabulate(get_basic_statistics(results),
                                     headers=["Exp.", "Sparsity", "Mean Acc.", "StDev", "Counts"])
    print(table_output)

# pairwise comparison
if 'c' in sys.argv[1]:
    exp_names = list(results.keys())
    first_row = ["Exp."] + exp_names
    table_raw = [first_row]
    for exp_name in exp_names:
        table_raw.append([exp_name] + ['/'] * (len(exp_names)))

    for expA in exp_names:
        i = exp_names.index(expA) + 1
        for expB in exp_names[i:]:
            j = exp_names.index(expB) + 1
            pv = scipy.stats.ttest_ind(results[expA][:, 1], results[expB][:, 1], equal_var=False).pvalue
            table_raw[i][j] = table_raw[j][i] = f"{pv:.5f}"

    print(tabulate.tabulate(table_raw, headers='firstrow'))
