from collections import Counter
from io import StringIO
import numpy as np
import numba
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


def digitize(s: str) -> list[npt.NDArray]:
    return [np.array([int(x) for x in l.split()]) for l in s.split("\n")]


def day_two_part_one(puzzle: str):
    lines = digitize(puzzle)
    cnt = 0
    for line in lines:
        if line.size == 0:
            continue
        d = np.diff(line)
        a = np.abs(d)
        strict = np.all(d > 0) or np.all(d < 0)
        atleast = np.all(a > 0)
        atmost = np.all(a <= 3)
        if strict and (atleast and atmost):
            cnt += 1
    return cnt


@numba.jit(nopython=True)
def increasing_with_dropout(a: npt.NDArray):
    assert len(a) > 2
    dropout = 0
    prev = 0
    sign = np.sign(prev)
    for i in range(1, len(a)):
        vnext = a[i] - a[prev]
        anext = abs(vnext)
        vsign = np.sign(anext)
        if anext == 0 or anext >= 3:
            dropout += 1
        if vsign != sign:
            # violated total order
            # drop the next value by not updating prev, this won't work in some edge cases where the thing we need to drop comes first.
            dropout += 1
        else:
            prev = i
            # need to update a[i - 1] in next iteration
            # instead of doing a[i - 1], manually update prev pointer
        sign = vsign


def day_two_part_two(puzzle: str):
    lines = digitize(puzzle)
    cnt = 0
    for line in lines:
        if line.size == 0:
            continue
        d = np.diff(line)
        a = np.abs(d)
        decr = np.where(d < 0)[0]
        incr = np.where(d > 0)[0]
        # check if there exists a number that is common to all of the cases that are violated, remove it
        # this won't check against the new list though, since we could have a transistion from 2-7-9 which
        # would still fail.
        if incr.size == line.size or decr.size == line.size:
            atleast = np.where(a == 0)[0]
            atmost = np.where(a > 3)[0]
        print(incr, decr, atleast, atmost)
        cnt += 1
    return cnt
