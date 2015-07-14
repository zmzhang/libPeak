import numpy as np
from scipy.stats import mode
cimport numpy as np
cimport cython
cimport libcpp.deque
cimport libcpp.map
cimport libcpp.vector
from libcpp cimport bool
from libcpp.deque  cimport *
from libcpp.map    cimport *
from libcpp.vector cimport *
from cython.operator cimport dereference as deref, preincrement as inc

cdef extern from "<algorithm>" namespace "std":
    void partial_sort[RandomAccessIterator](RandomAccessIterator first, RandomAccessIterator middle, RandomAccessIterator last)


cpdef p_sort():
    cdef vector[int] v
    cdef int i = 0
    cdef list res = []
    v.push_back(4)
    v.push_back(6)
    v.push_back(2)
    v.push_back(5)
    partial_sort[vector[int].iterator](v.begin(), v.end(), v.end())
    for i in v:
        res.append(i)
    return res
 
@cython.boundscheck(False)
@cython.wraparound(False)
def _ridge_detection(np.ndarray[np.uint8_t, cast=True, ndim=2] local_max, int row_best, int col,
                     int n_rows, int n_cols, int minus=True, int plus=True):
 
    cdef libcpp.deque.deque[int] cols = deque[int]()
    cdef libcpp.deque.deque[int] rows = deque[int]()
    cdef int col_plus = col
    cdef int col_minus = col
    cdef int segment_plus = 1
    cdef int segment_minus = 1
    cdef int row_plus, row_minus, i
 
    cols.push_back(col)
    rows.push_back(row_best)
 
    for i in range(1, n_rows):
        row_plus = row_best + i
        row_minus = row_best - i
        if minus and row_minus > 0 and segment_minus < col_minus < n_cols - segment_minus - 1:
            if local_max[row_minus, col_minus + 1]:
                col_minus += 1
            elif local_max[row_minus, col_minus - 1]:
                col_minus -= 1
            elif local_max[row_minus, col_minus]:
                col_minus = col_minus
            else:
                col_minus = -1
            if col_minus != -1:
                rows.push_front(row_minus)
                cols.push_front(col_minus)
        if plus and row_plus < n_rows and segment_plus < col_plus < n_cols - segment_plus - 1:
            if local_max[row_plus, col_plus + 1]:
                col_plus += 1
            elif local_max[row_plus, col_plus - 1]:
                col_plus -= 1
            elif local_max[row_plus, col_plus]:
                col_plus = col_plus
            else:
                col_plus = -1
            if col_plus != -1:
                rows.push_back(row_plus)
                cols.push_back(col_plus)
        if (minus and False == plus and col_minus == -1) or \
                (False == minus and True == plus and col_plus == -1) or \
                (True == minus and True == plus and col_plus == -1 and col_minus == -1):
            break
    return [rows[i] for i in range(rows.size())], [cols[i] for i in range(cols.size())]
 
@cython.boundscheck(False)
@cython.wraparound(False)
cdef _local_extreme(data, comparator,
                  axis=0, order=1, mode='clip'):
    if (int(order) != order) or (order < 1):
        raise ValueError('Order must be an int >= 1')
    datalen = data.shape[axis]
    locs = np.arange(0, datalen)
    results = np.ones(data.shape, dtype=np.bool)
    main = data.take(locs, axis=axis, mode=mode)
    for shift in xrange(1, order + 1):
        plus = data.take(locs + shift, axis=axis, mode=mode)
        minus = data.take(locs - shift, axis=axis, mode=mode)
        results &= comparator(main, plus)
        results &= comparator(main, minus)
        if ~results.any():
            return results
    return results


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline _mode(np.ndarray[np.int_t, cast=True, ndim=2] ridge, np.ndarray[np.float64_t, ndim=2] cwt2d):
    cdef np.int i, key, t, m
    cdef np.long n = ridge.shape[1]

    cdef libcpp.map.map[int,libcpp.vector.vector[int]]   counts = map[int,libcpp.vector.vector[int]]()
    for i in range(n):
        if cwt2d[ridge[0,i], ridge[1,i]] > 0:
            counts[ridge[1,i]].push_back(ridge[0,i])
    cdef map[int,libcpp.vector.vector[int]].iterator im = counts.begin()
    m = 0
    while im != counts.end():
        t = deref(im).second.size()
        if t > m :
            m = t
            key = deref(im).first
        inc(im)
    if m > 0:
        return key,counts[key][0]
    else:
        return -1, -1

@cython.boundscheck(False)
@cython.wraparound(False)
def _peaks_position(np.ndarray[np.float64_t, ndim=1] vec, ridges,
                        np.ndarray[np.float64_t, ndim=2] cwt2d):
    cdef int n_cols = cwt2d.shape[1], n_rows = cwt2d.shape[0]
    cdef int n_ridges = len(ridges)
    cdef int i, j
    cdef int row, col, cols_start, cols_end, max_ind, ridge_ind
    cdef double max_val
    cdef libcpp.vector.vector[int] peaks
    cdef libcpp.vector.vector[int] ridges_select
    cdef libcpp.vector.vector[int] rows


    cdef np.ndarray[np.uint8_t, cast=True, ndim=2] negs = cwt2d < 0
    cdef np.ndarray[np.uint8_t, cast=True, ndim=2] local_minus = _local_extreme(cwt2d, np.less, axis=1, order=1)
    negs[:, [0, n_cols - 1]] = True
    negs |= local_minus

    for ridge_ind in range(n_ridges):
        col, row = _mode(ridges[ridge_ind], cwt2d)
        if row > 0:
            cols_start = -n_cols
            cols_end = -n_cols
            for i in range(n_cols):
                if cols_start == -n_cols and negs[row, col - i]:
                    cols_start = col - i + 1
                if cols_end == -n_cols and negs[row, col + i]:
                    cols_end = col + i
                if cols_end != -n_cols and cols_start != -n_cols:
                    break
            max_ind = -1
            max_val = -10e20
            # print col, row, cols_start, cols_end
            for i in range(cols_start , cols_end):
                if vec[i] > max_val:
                    max_val = vec[i]
                    max_ind = i
            peaks.push_back(max_ind)
            ridges_select.push_back(ridge_ind)
        elif ridges[ridge_ind].shape[1] > 2: # local wavelet coefficients < 0
            cols_accurate = ridges[ridge_ind][1, 0:ridges[ridge_ind].shape[1] / 2]
            cols_start = max(np.min(cols_accurate) - 3, 0)
            cols_end = min(np.max(cols_accurate) + 4, n_cols - 1)
            max_ind = -1
            max_val = -10e20
            for i in range(cols_start , cols_end):
                if vec[i] > max_val:
                    max_val = vec[i]
                    max_ind = i
            peaks.push_back(max_ind)
            ridges_select.push_back(ridge_ind)


    cdef libcpp.vector.vector[int] peaks_refine
    cdef libcpp.vector.vector[int] ridges_refine
    cdef int n = peaks.size()
    ridges_len = [ridges[i].shape[1] for i in ridges_select]

    cdef int peak_len = ridges_len[0]
    cdef int peak = peaks[0]
    cdef int peak_ind = 0
    for i in range(1, n):
        print peaks[i], ridges_len[i]
        if peaks[i] == peak and ridges_len[i] >= peak_len:
            peak_len = ridges_len[i]
            peak = peaks[i]
            peak_ind = i
        if peaks[i] != peak:
            if (vec[peak] > vec[max(peak - 1, 0)] or vec[peak] > vec[max(peak - 2, 0)]) \
                    and (vec[peak] > vec[min(peak + 1, n_cols)] or vec[peak] > vec[min(peak + 2, n_cols)]):
                peaks_refine.push_back(peak)
                ridges_refine.push_back(ridges_select[peak_ind])
            peak = peaks[i]
            peak_ind = i
        if i == n - 1:
            if (vec[peak] > vec[max(peak - 1, 0)] or vec[peak] > vec[max(peak - 2, 0)]) \
                    and (vec[peak] > vec[min(peak + 1, n_cols)] or vec[peak] > vec[min(peak + 2, n_cols)]):
                peaks_refine.push_back(peak)
                ridges_refine.push_back(ridges_select[peak_ind])
    return [peaks_refine[i] for i in range(peaks_refine.size())], \
           [ridges[int(ridges_refine[i])] for i in range(ridges_refine.size())]