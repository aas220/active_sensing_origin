#!/usr/bin/python

import sys
import numpy as np


# 95% confidence interval.
Z = 1.96


def confidence_interval(a, z):
    mean = np.mean(a)
    std = np.std(a)
    dev = z * std / np.sqrt(float(len(a)))
    return mean, dev


if __name__ == '__main__':
    file_path = sys.argv[1]
    file = open(file_path, 'r')
    method = None
    random_rewards = []
    state_rewards = []
    action_rewards = []

    for line in file:

        if line.startswith('Random Active Sensing Simulations'):
            method = 'random'
            continue

        elif line.startswith('State-Entropy Active Sensing Simulations'):
            method = 'state'
            continue

        elif line.startswith('Action-Entropy Active Sensing Simulations'):
            method = 'action'
            continue

        elif line.startswith('average reward'):
            method = None
            continue

        split_line = line.split(' ')

        if method is not None:

            is_successful = int(split_line[12])

            if method == 'random' and is_successful == 1:
                random_rewards.append(float(split_line[4][0:-1]))

            elif method == 'state' and is_successful == 1:
                state_rewards.append(float(split_line[4][0:-1]))

            elif method == 'action' and is_successful == 1:
                action_rewards.append(float(split_line[4][0:-1]))

    z = 1.96
    print confidence_interval(random_rewards, z)
    print confidence_interval(state_rewards, z)
    print confidence_interval(action_rewards, z)

    file.close()

