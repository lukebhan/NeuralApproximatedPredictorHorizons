import numpy as np

# x(tq) is given by 
# x_history(tq) for tq <= 0
# linear interpolation of x_hst if 0 < tq < t_cur
# x_hst[-1] if tq>t_cur
def interp_history(tq, t_hist, x_hist, history_fn):
    if tq <= 0.0:
        return np.asarray(history_fn(tq), dtype=float).reshape(-1)

    if tq >= t_hist[-1]:
        return x_hist[-1].copy()

    j = np.searchsorted(t_hist, tq) - 1
    j = max(0, min(j, len(t_hist) - 2))

    t0, t1 = t_hist[j], t_hist[j + 1]
    x0, x1 = x_hist[j], x_hist[j + 1]
    a = (tq - t0) / (t1 - t0)
    return (1 - a) * x0 + a * x1

def set_size(width, fraction=1, subplots=(1, 1), height_add=0):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = height_add + fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)