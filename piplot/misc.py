import numpy as np
import warnings

from IPython.display import display
import ipywidgets as ipw

import matplotlib.patches


import pandas as pd
from pandas import DataFrame

import numbers

import scipy.stats as st

try:
    import plotly.graph_objs as go
except ModuleNotFoundError as err:
    print("Warning: plotly not installed.")

import matplotlib.pyplot as pl

def get_plotly_Surface_from_fun(x, y, fun, **kwargs):
    z = np.array([  [
            fun(xx, yy)
            for xx in x
        ]
        for yy in y
    ])
    
    return go.Surface(
        x = x,
        y = y,
        z = z,
        **kwargs
    )
    
def get_plotly_Surface_from_interpolator(fun, nx = 30, ny = 30, **kwargs):
    assert len(fun.params) == 2, "Dimension of the domain must be 2."
    
    (xmin, ymin), (xmax, ymax) = fun.get_range_of_params()

    return get_plotly_Surface_from_fun(
        x = np.linspace(xmin, xmax, nx),
        y = np.linspace(ymin, ymax, ny),
        fun = lambda x,y : fun([x,y]),
        **kwargs
    )

def plot_2d_transf(x,y, **kwargs):
    """
    x, y are np.ndarrays of reals
    """
    #pl.axes().set_aspect('equal', 'datalim')
    m, n = x.shape
    if "c" not in kwargs:
        kwargs["c"] = "grey"
    for i in range(n):
        pl.plot( x[:,i], y[:,i], **kwargs)
    for i in range(m):
        pl.plot( x[i,:], y[i,:], **kwargs)
        
#def plot_interpolator_heatmap(f, nx = 30, ny = 30 ):
#    (xmin, ymin), (xmax, ymax) = f.get_range_of_params()
#    return plot_fun_heatmap(
#        f,
#        xxx = np.linspace(xmin, xmax, nx),
#        yyy = np.linspace(ymin, ymax, ny),
#    )
    

    
def plot_fun_heatmap(f, xxx, yyy):
    fff = np.array([[ f([x,y]) for y in yyy ] for x in xxx])

    cp = pl.contourf( xxx, yyy, fff.T, cmap=pl.cm.rainbow)
    
    return cp
    
def color_2D_data(x, y):
    """Assign colors to dots depending on two features. Useful in scatterplots.
    
    Args:
        x, y: real-valued pandas Series with common index.
        
    Returns:
        DataFrame with the same index as `x, y` and columns `r, g, b, hex`.
        The `hex` column can be passed directly to the `matplotlib.scatter`.    
    """
    ## coloring the rows from bookmaker's data
    def rgb_to_hex_color(r, g, b):
        """r, g, b -- real numbers supposed to be in the interval [0, 1]"""
        cc = [r, g, b]

        def rescale(c):
            if np.isnan(c):
                return 0
            assert 0 <= c <= 1
            return int(round(c * 255 ))

        cc = [ rescale(c)  for c in cc]

        return '#%02x%02x%02x' % tuple(cc)


    def normalize(xx):
        xx_min = xx[~np.isnan(xx)].min()
        xx_max = xx[~np.isnan(xx)].max()
        
        return (xx - xx_min) / (xx_max - xx_min)
        
    cc = DataFrame(
        data = [ normalize(c) for c in [x, y]],
        index = ["b", "g"]
    ).T
                   
    cc["r"] = ( 1 - (cc.b+ cc.g)/2)
    cc["hex"] = cc.apply( lambda x: rgb_to_hex_color(x.r, x.g, x.b), axis = 1)
    
    return cc
    
from matplotlib.patches import Ellipse

def plot_distr_ellipse(mean, cov, ax = None, color = "black"):
#     mean = np.array(mean)
#     cov = np.array(cov)
    
    if ax is None:
        ax = pl.gca()
    
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    ell = Ellipse(
        xy=mean,
        width=lambda_[0]*2, height=lambda_[1]*2,
                  angle=np.rad2deg(np.arccos(v[0, 0])))
    ell.set_facecolor('none')
    ell.set_edgecolor(color)
    
    if ax is None:
        ax = pl.gca()
    ax.add_artist(ell)
    
    
## Courtesy of Josef "Pepa" Ondrej
def picker_text(xs, ys, f_index,
    fig_width = 5, fig_height = 5, text_width = "500px", text_height = "300px"):
    """
    Creates a scatterplot of xs,ys and textfield. User can select point in the
    scatterplot and in the textfield appears text f_index(i), where i is index
    of the point -- point with coordinates [xs[16], ys[16]] has index 16

    Args:
    xs -- A np.array of floats, x coordinates of points
    ys -- A np.array of floats, y coordinates of points
    f_index -- A function that gets a `int` as argument and returns string

    Returns:
    --
    """
    def onclick(event):
        x = event.mouseevent.xdata
        y = event.mouseevent.ydata

        distances = np.hypot(x - xs, y - ys)
        ind = distances.argmin()

        ax.cla()
        ax.scatter(xs, ys, picker=30, alpha=0.5, s=50)
        ax.scatter(xs[ind], ys[ind], c="red", alpha=0.5, s=200)

        textar.value=(str(f_index(ind)))

    fig, ax = pl.subplots(1,1,figsize = (fig_width, fig_height))
    ax.scatter(xs, ys, picker=30, alpha=.5, s=50)
    fig.canvas.mpl_connect('pick_event', onclick)

    textar = ipw.Textarea(value="Index: --", layout=ipw.Layout(width='500px', height='150px'))
    display(textar)
    
#### Example (run in jupyter notebook)
#import matplotlib.pyplot as pl
#%matplotlib notebook
#import pi_plotting_module_02 as pipl
#
#import numpy.random as rnd
#
#xs, ys = rnd.rand(2, 20)
#    
#pipl.picker_text(
#    xs, ys, 
#    f_index=lambda i: "point_{} = [{:.2f}, {:.2f}]".format(i, xs[i], ys[i])
#) 



def plot_discrete_model_vs_data_histogram(
    probas, obs, normed = False, 
    color = "grey", ecolor = "red",
    err_bar_proba = 0.96
):
    """
    probas: array with shape [n_obs, n_bars]
    obs: integer array with shape [n_obs] satisfying 0 <= obs < n_bars.
    normed: boolean indicating whether each bar will be multiplied by 
        the model-proba concentrated in the coresponding bin.
    color: color of the bars
    ecolor: color of error-bar
    err_bar_proba: real number between 0 and 1. Larger means larger error bar.
    """

    n_obs, n_bars = probas.shape 
    
    model_y = probas.sum(axis = 0)
    model_std =  np.sum(
        probas * (1 - probas),
        axis = 0
    )**(1/2)
        
    obs_y = pd.value_counts(obs).reindex(np.arange(n_bars)).fillna(0)
    
    if normed:
        model_std = model_std / model_y
        obs_y = obs_y / model_y
        model_y = model_y / model_y
    
    bar_width = 1

    err_bar_coef = st.norm().ppf( 1 - (1 - err_bar_proba)/2)

    ## the bars of the plot will represent observations
    pl.bar(
        x = np.arange(n_bars), #+ bar_width,
        height = obs_y,
        width = bar_width,
        color = color,
        edgecolor = "none"
    )
    
    ## From the model we keep visible only the orror bars
    pl.bar(
        x = np.arange(n_bars),
        height = model_y,
        width = bar_width,
        yerr = model_std * err_bar_coef, 
        color = "none",
        #edgecolor = "none",
        ecolor = ecolor
    )
    
    
def plot_model_quants_vs_data_histogram(
    pdfs, obs, normed = True, 
    color = "grey", ecolor = "red",
    quant_bins = 10,
    err_bar_proba = 0.96
):
    """
    pfs: array with shape [n_obs, max_X]
    obs: integer array with shape [n_obs] satisfying 0 <= obs < max_X.
    normed: boolean indicatig whether each bar will be multiplied by 
        the model-proba concentrated in the coresponding bin.
    color: color of the bars
    ecolor: color of error-bar
    quant_bins: either an integer or a 1d array starting with 0, ending with 1.
    err_bar_proba: real number between 0 and 1. Larger means larger error bar.
    """
    
    if isinstance(quant_bins, numbers.Number):
        quant_bins = np.linspace(0, 1, quant_bins +1, endpoint=True)

    eps = 0.1
    quant_bins[0] = -eps
    quant_bins[-1] = 1 + eps
    
    n_obs, max_X = pdfs.shape
    n_bins = len(quant_bins) -1

    ## our cummulative distribution functions `cdfs`  will count the 
    ## boundary with coefficient 1/2 (not 1 as usual)
    cdfs = pdfs.cumsum(axis = -1) - pdfs/2
    if not (-eps < cdfs).all():
        warnings.warn(
            f"(in method plot_model_quants_vs_data_histogram) "
            "cdf shoud be between 0, 1. The minimum {cdfs.min()} "
            "is attained for the batch-index {cdfs.min(axis = 1).argmin()}."
        )

    if not (cdfs < 1 + eps).all():
        warnings.warn(
            f"(in method plot_model_quants_vs_data_histogram) "
            "cdf shoud be between 0, 1. The maximum {cdfs.max()} "
            "is attained for the batch index {cdfs.max(axis = 1).argmax()}."
        )
        
    cdfs = np.maximum(0, np.minimum(cdfs, 1))
    
    def cut_array(x, bins):
        """The same thing as pandas.cut, only accepts higher dimensional arrays.
        (pandas.cut accepts only one-dimesional x)
        """
        sh = x.shape
        return np.reshape(
            pd.cut(x.flat, bins, labels = False),
            newshape = sh
        )

    bin_indicators = cut_array(cdfs, quant_bins)
    obs_bins = cut_array(cdfs[np.arange(n_obs), obs], quant_bins)

    assert not np.any(np.isnan(bin_indicators))
    
    bin_probas = np.zeros([n_obs, n_bins])
    for i in range(n_obs):
        for x in range(max_X):
            bin_probas[i, bin_indicators[i, x]] += pdfs[i, x]

    plot_discrete_model_vs_data_histogram(
        bin_probas, obs_bins, normed = normed,
        color = color, ecolor = ecolor,
        err_bar_proba = err_bar_proba
    )

def draw_rectangle(min_xy, max_xy, **kwargs):
    min_xy, max_xy = [np.array(v) for  v in [min_xy, max_xy]]
    dif = max_xy - min_xy
    pl.gca().add_patch(
        matplotlib.patches.Rectangle(
            min_xy,   # (x,y)
            dif[0],          # width
            dif[1],          # height
            **kwargs
        )
    )