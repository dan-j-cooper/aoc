from collections import Counter
from io import StringIO
import numpy as np
import numpy.typing as npt


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
    mat = np.loadtxt(StringIO(puzzle))
    difference = np.abs(np.diff(mat, axis=-1))
    safe_delta = np.all(difference >= 1, axis=-1) & np.all(difference <= 3, axis=-1)
    sorted_mat = np.sort(mat, axis=-1)
    reverse_sort = sorted_mat[..., ::-1]
    ordered = np.all(mat == sorted_mat, axis=-1) | np.all(mat == reverse_sort, axis=-1)
    total = np.logical_and(safe_delta, ordered)

    return np.sum(total)


def _increasing(vec: npt.NDArray):
    return np.apply_over_axes()


def day_two_part_two(puzzle: str):
    pass
