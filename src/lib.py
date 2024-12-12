from collections import Counter
import numpy as np


def day_one_part_one(puzzle: str):
    vleft, vright = [], []
    for row in puzzle.splitlines():
        left, right = row.split()
        vleft.append(int(left))
        vright.append(int(right))

    vleft = np.array(vleft)
    vright = np.array(vright)
    vleft.sort()
    vright.sort()
    return sum(np.abs(vleft - vright))


def day_one_part_two(puzzle: str):
    vleft, vright = [], []
    for row in puzzle.splitlines():
        left, right = row.split()
        vleft.append(int(left))
        vright.append(int(right))

    cright = Counter(vright)
    for i, v in enumerate(vleft):
        vleft[i] = v * cright[v]

    return sum(vleft)

def day_two_part_one(puzzle: str):
