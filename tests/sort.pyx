import random


def sort(list arr_1):
    cdef int l, j, i
    l = len(arr_1)
    for i in range(l):
        for j in range(l):
            if arr_1[i] > arr_1[j]:
                arr_1[i], arr_1[j] = arr_1[j], arr_1[i]


def main():
    cdef list arr
    cdef int n = 1000
    arr = [random.randint(0, 100) for i in range(n)]
    sort(arr)


