from __future__ import division
 
import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import scoreatpercentile, mode
from collections import deque
#from _peak_detection import _ridge_detection, _peaks_position

from pylab import plot, imshow, show, figure, bar, legend, xlabel, ylabel
 

from matplotlib import colors
cmap_red = colors.ListedColormap([[1.0,1.0,1.0,0.0], [1.0,0.0,0.0,1.0]])
bounds = [0, 0.5, 1]
norm = colors.BoundaryNorm(bounds, cmap_red.N)

cmap_blue = colors.ListedColormap([[1.0,1.0,1.0,0.0], [0.0,1.0,0.0,1.0]])
bounds = [0, 0.5, 1]
norm = colors.BoundaryNorm(bounds, cmap_blue.N)

cmap_black = colors.ListedColormap([[1.0,1.0,1.0,0.0], [0.0,0.0,1.0,1.0]])
bounds = [0, 0.5, 1]
norm = colors.BoundaryNorm(bounds, cmap_black.N)

def mexican_hat(points, a):
    A = 2 / (np.sqrt(3 * a) * (np.pi ** 0.25))
    wsq = a ** 2
    vec = np.arange(0, points) - (points - 1.0) / 2
    tsq = vec ** 2
    mod = (1 - tsq / wsq)
    gauss = np.exp(-tsq / (2 * wsq))
    total = A * mod * gauss
    return total
 
 
def cwt(data, wavelet, widths):
    output = np.zeros([len(widths), len(data)])
    for ind, width in enumerate(widths):
        wavelet_data = wavelet(min(10 * width, len(data)), width)
        output[ind, :] = fftconvolve(data, wavelet_data,
                                     mode='same')
    return output
 
 
def local_extreme(data, comparator,
                  axis=0, order=1, mode='clip'):
    if (int(order) != order) or (order < 1):
        raise ValueError('Order must be an int >= 1')
    datalen = data.shape[axis]
    locs = np.arange(0, datalen)
    results = np.ones(data.shape, dtype=bool)
    main = data.take(locs, axis=axis, mode=mode)
    for shift in range(1, order + 1):
        plus = data.take(locs + shift, axis=axis, mode=mode)
        minus = data.take(locs - shift, axis=axis, mode=mode)
        results &= comparator(main, plus)
        results &= comparator(main, minus)
    return results
 
 
def ridge_detection(local_max, row_best, col, n_rows, n_cols, minus=True, plus=True):
    cols = deque()
    rows = deque()
    cols.append(col)
    rows.append(row_best)
    col_plus = col
    col_minus = col
    for i in range(1, n_rows):
        row_plus = row_best + i
        row_minus = row_best - i
        segment_plus = 1
        segment_minus = 1
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
                rows.appendleft(row_minus)
                cols.appendleft(col_minus)
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
                rows.append(row_plus)
                cols.append(col_plus)
        if (minus and False == plus and col_minus == -1) or \
                (False == minus and True == plus and col_plus == -1) or \
                (True == minus and True == plus and col_plus == -1 and col_minus == -1):
            break
    return rows, cols
 
 
def peaks_position(vec, ridges, cwt2d, wnd=2):
    n_cols = cwt2d.shape[1]
    negs = cwt2d < 0
    local_minus = local_extreme(cwt2d, np.less, axis=1, order=1)

    zero_crossing = np.abs(np.diff(np.sign(cwt2d))) / 2
    # # figure(figsize=(12, 3))
    # imshow(zero_crossing, cmap=cmap_black)
    # ylabel("Scales")
    # # figure(figsize=(12, 3))
    # imshow(local_minus, cmap=cmap_blue)
    # ylabel("Scales")

    negs |= local_minus
    negs[:, [0, n_cols - 1]] = True
    ridges_select = []
    peaks = []
 
    for ridge in ridges:
        inds = np.where(cwt2d[ridge[0, :], ridge[1, :]] > 0)[0]
        if len(inds) > 0:
            col = int(mode(ridge[1, inds])[0][0])
            rows = ridge[0, :][(ridge[1, :] == col)]
            row = rows[0]
            cols_start = max(col - np.where(negs[row, 0:col][::-1])[0][0], 0)
            cols_end = min(col + np.where(negs[row, col:n_cols])[0][0], n_cols)
            # print col, row, cols_start, cols_end
            if cols_end > cols_start:
                inds = range(cols_start, cols_end)
                peaks.append(inds[np.argmax(vec[inds])])
                ridges_select.append(ridge)
        elif ridge.shape[1] > 2: # local wavelet coefficients < 0
            cols_accurate = ridge[1, 0:int(ridge.shape[1] / 2)]
            cols_start = max(np.min(cols_accurate) - 3, 0)
            cols_end = min(np.max(cols_accurate) + 4, n_cols - 1)
            inds = range(cols_start, cols_end)
            if len(inds) > 0:
                peaks.append(inds[np.argmax(vec[inds])])
                ridges_select.append(ridge)
    # print peaks
    ridges_refine = []
    peaks_refine = []
    ridges_len = np.array([ridge.shape[1] for ridge in ridges_select])
    # print zip(peaks, ridges_len)
    for peak in np.unique(peaks):
        inds = np.where(peaks == peak)[0]
        ridge = ridges_select[inds[np.argmax(ridges_len[inds])]]
        inds = np.clip(range(peak - wnd, peak + wnd + 1), 0, len(vec) - 1)
        inds = np.delete(inds, np.where(inds == peak))
        if np.all(vec[peak] > vec[inds]):
            ridges_refine.append(ridge)
            peaks_refine.append(peak)
    return peaks_refine, ridges_refine
 
 
def ridges_detection(cwt2d, vec):
    n_rows = cwt2d.shape[0]
    n_cols = cwt2d.shape[1]
    local_max = local_extreme(cwt2d, np.greater, axis=1, order=1)
    ridges = []
    rows_init = np.array(range(1, 6))
    cols_small_peaks = np.where(np.sum(local_max[rows_init, :], axis=0) > 0)[0]
    for col in cols_small_peaks:
        best_rows = rows_init[np.where(local_max[rows_init, col])[0]]
        rows, cols = ridge_detection(local_max, best_rows[0], col, n_rows, n_cols, True, True)
        staightness = 1 - float(sum(abs(np.diff(cols)))) / float(len(cols))
        if len(rows) >= 2 and \
            staightness > 0.2 and  \
            not(
            len(ridges) > 0 and
            rows[0] == ridges[-1][0, 0] and
            rows[-1] == ridges[-1][0, -1] and
            cols[0] == ridges[-1][1, 0] and
            cols[-1] == ridges[-1][1, -1] and
            len(rows) == ridges[-1].shape[1]
        ):
            ridges.append(np.array([rows, cols], dtype=np.int32))

    # figure(figsize=(12, 3))
    # imshow(cwt2d)
    # ylabel("Scales")
#    figure()
#    plot(cwt2d[3,:])
#     figure(figsize=(9, 4))
#     imshow(local_max, cmap=cmap_red)
#     ylabel("Scales")

    return ridges
 
 
def signal_noise_ratio(cwt2d, ridges, peaks):
    n_cols = cwt2d.shape[1]
    row_one = cwt2d[0, :]
    row_one_del = np.delete(row_one, np.where(abs(row_one) < 10e-5))
    t = 3 * np.median(np.abs(row_one_del - np.median(row_one_del))) / 0.67
    row_one[row_one > t] = t
    row_one[row_one < -t] = -t
    noises = np.zeros(len(peaks))
    signals = np.zeros(len(peaks))
    for ind, val in enumerate(peaks):
        hf_window = ridges[ind].shape[1] * 1
        window = range(int(max([val - hf_window, 0])), int(min([val + hf_window, n_cols])))
        noises[ind] = scoreatpercentile(np.abs(row_one[window]), per=90)
        signals[ind] = np.max(cwt2d[ridges[ind][0, :], ridges[ind][1, :]])
    sig = [1 if s > 0 and n >= 0 else - 1 for s, n in zip(signals, noises)]
 
    # print zip(peaks, signals, noises)
    # figure()
    # plot(row_one, label='scale = 1')
    # # plot(cwt2d[1, :], label='scale = 2')
    # plot(cwt2d[3, :], label='scale = 4')
    # # plot(cwt2d[5, :], label='scale = 6')
    # legend()
    return np.sqrt(np.abs((signals + np.finfo(float).eps) / (noises + np.finfo(float).eps))) * sig, signals
 
 
def peaks_detection(vec, scales, min_snr=3):
    cwt2d = cwt(vec, mexican_hat, scales)
    ridges = ridges_detection(cwt2d, vec)
    # print ridges
    peaks, ridges = peaks_position(vec, ridges, cwt2d)
    # print ridges
    # print peaks
    snr, signals = signal_noise_ratio(cwt2d, ridges, peaks)
    # print zip(peaks, snr)
    # peaks_refine = [peak for i, peak in enumerate(peaks) if snr[i] >= min_snr]
    # signals_refine = [signal for i, signal in enumerate(signals) if snr[i] >= min_snr]
    # print peaks_refine
    peaks_refine = [peak for i, peak in enumerate(peaks) if signals[i] >= min_snr]
    signals_refine = [signal for i, signal in enumerate(signals) if signals[i] >= min_snr]
    return peaks_refine, signals_refine


if __name__ == '__main__':

    from scipy.stats import norm
    import matplotlib.pyplot as plt
    
    cs=[(255,158,74),
    (237,102,93),
    (173,139,201),
    (114,158,206),
    (103,191,92),
    (237,151,202),
    (205,204,93),
    (168,120,110),
    (162,162,162),
    (109,204,218)]
    
    csn=[tuple([i/255.0 for i in t]) for t in cs]
    lw=2.0
    
    size = 200
    xa = np.arange(0, size, 1)
    noise = np.random.normal(0,1,size)
    baseline = xa*0.1
    rv1 = norm(loc = 100, scale = 5)
    rv2 = norm(loc = 50, scale = 3)
    rv3 = norm(loc = 150, scale = 10)
    x=(rv1.pdf(xa)+rv2.pdf(xa)+rv3.pdf(xa))*500 + noise + baseline
    
    peak_ind, sig = peaks_detection(x, np.arange(1, 30), 50)
    
    plt.figure(figsize=(9, 3))
    plt.plot(xa, x, linewidth=lw,color=csn[0])
    plt.plot(xa[peak_ind], x[peak_ind], 'bv', markersize=10)
    plt.ylabel("Intensity")
    
    plt.show()