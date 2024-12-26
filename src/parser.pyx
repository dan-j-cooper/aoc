import cython
cdef extern from "<map>" import unordered_map


cdef unordered_map[string, int] hmap

def test():
    cdef hmap cnts
    cnts[b"hi"] = 1
    return cnts[b"hi"]

