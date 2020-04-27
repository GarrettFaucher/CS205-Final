import numpy as np

import pandas as pd
from crontab import Crontab

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch
import matplotlib.patheffects as path_effects

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from make_data import *




ROSE_CMAP = 'cividis'

# Does a fit of (X,Y) to x_fit using using linear regression gradient descent
def fit(X,Y,x_fit):
    model = LinearRegression()
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(np.array(X).reshape(-1,1))
    x_fit_poly = poly.fit_transform(np.array(x_fit).reshape(-1,1))
    model.fit(X_poly,Y)
    return model.predict(x_fit_poly)

# Generates the plot of temp/pressure/windspeed against time
def stack_plot(data, forecast=False, depth=.3, save=False):
    min_time = min(data['time'])
    max_time = max(data['time'])

    # If we are attempting a forecast, extend the max time beyond the provided data
    if forecast:
        max_time = max_time + depth*(max_time - min_time)/2.

    fig,ax = plt.subplots(ncols = 1, nrows = 3,
                          sharex = True,
                          figsize=(10,10),
                          subplot_kw=dict(xlim=(min_time,max_time)),
                          gridspec_kw=dict(hspace=0))

    # kwargs to pass on data plots
    plot_kwargs = dict(color='k',
                       linestyle='-',
                       linewidth=2)

    outline = [path_effects.Stroke(linewidth=1, foreground='black'),
               path_effects.Normal()]

    # kwargs to pass to text objects for the black outline
    text_kwargs = dict(color='w',
                       path_effects=outline)

    # Plotting temperature
    ax[0].plot(data['time'], data['temp'], **plot_kwargs)
    ax[0].set_ylabel(r"Temperature ($^\circ$C)", fontsize=14, **text_kwargs)
    ax[0].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    # Plotting pressure
    ax[1].plot(data['time'], data['pressure'], **plot_kwargs)
    ax[1].set_ylabel("Pressure (atm)", fontsize=14, **text_kwargs)
    ax[1].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2f}'))

    # Plotting wind speed
    ax[2].plot(data['time'], data['windspeed'], **plot_kwargs)
    ax[2].set_ylabel("Wind speed (mph)", fontsize=14, **text_kwargs)
    ax[2].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2f}'))

    # Setting time ticks and labelling x-axis
    x_ticks = np.arange(min_time, max_time, 1)
    ax[2].set_xticks(x_ticks)

    # Adding outline to xtick labels
    for tick in ax[2].get_xticklabels():
        tick.set_path_effects(outline)
        tick.set_color('white')
    ax[2].set_xlabel("Time (sol)", fontsize=16, labelpad=10, **text_kwargs)

    # kwargs to pass to grid lines
    tickline_kwargs = dict(color='k',
                           linestyle='--',
                           linewidth=.5,
                           alpha=.6)
    # Vertical grid lines
    for x in x_ticks:
        for a in ax:
            a.axvline(x, **tickline_kwargs)

    s = .2 # Scale to set margin between data points and borders of graphs
    for col in data.columns[1:]:
        n = data.columns.get_loc(col) - 1
        min_col = min(data[col])
        max_col = max(data[col])

        # Adding margin
        delta = s/2.*(max_col - min_col)
        min_yax = min_col - delta
        max_yax = max_col + delta

        # Setting y-ticks for this ax
        y_ticks = [linearmap([0, 4],[min_yax,max_yax],i) for i in range(0,4)]
        ax[n].set_yticks(y_ticks)
        for tick in ax[n].get_yticklabels():
            tick.set_path_effects(outline)
            tick.set_color('white')

        # Horizontal grid lines
        for y in y_ticks:
            ax[n].axhline(y, **tickline_kwargs)
            ax[n].set_ylim(min_yax, max_yax)

        # Setting background color
        ax[n].patch.set_facecolor('k')
        ax[n].patch.set_alpha(.2)


    # Attempt to forecast data
    if forecast:
        # Setup
        N = len(data['time'])

        # Timescale over which we are doing a forecast
        t = np.linspace(data['time'].iloc[-len(data['time'])//6], max_time, 100)

        # For each variable, do a curve fit and plot uncertainty regions
        for n,col in enumerate(data.columns[1:]):
            fit_data = fit(data['time'], data[col], t)
            scale = .1*max(data[col] - min(data[col]))
            ax[n].fill_between(t, fit_data - scale*(t[0] - t), fit_data + scale*(t[0] - t),
                               facecolor='r',
                               alpha=.3)
            ax[n].plot(t, fit_data, 'r--')

    fig.align_ylabels()
    fig.patch.set_alpha(0.)

    # Either save or display the figure
    if save:
        plt.savefig("stackplot.png", format='png', facecolor=fig.get_facecolor())
    else:
        plt.show()

    plt.close()

# Make wind rose plot
def windrose(data, num_days=None, save=False):
    if num_days is not None:
        today = max(data['sol'])
        data = data[data['sol'] > today - num_days]

    fig, ax = plt.subplots(nrows=1, ncols=1,
                           figsize=(10,10),
                           subplot_kw=dict(projection='polar'))

    # Loading data (and converting winddirection to radians)
    wdir = np.pi/180.*data['winddirection']
    wcount = data['windcount']

    # Setting bin parameters
    N = len(wcount)
    ndirbins = 8
    width = 2*np.pi/ndirbins
    dirbins = np.arange(0, 2*np.pi, width)

    ncountbins = 4
    countbins = np.linspace(0, max(wcount), ncountbins+1)
    cmap = plt.cm.get_cmap(ROSE_CMAP)

    # Iterate backwards through countbins
    for n in np.flip(range(1, ncountbins+1)):
        count = countbins[n]

        # Binning data
        counts, bins = np.histogram(wdir[wcount < count], bins=dirbins)
        radii = 100.*counts/N

        # Plotting data
        ax.bar(bins[:-1], radii, align='edge', color=cmap((n-1)/(ncountbins-1)), width=width)

    outline = [path_effects.Stroke(linewidth=1, foreground='black'),
               path_effects.Normal()]

    # kwargs to pass to text objects for the black outline
    text_kwargs = dict(color='w',
                       path_effects=outline)

    # Setting tick properties
    ax.set_xticks(dirbins)
    dirticks = ["E", "ENE", "NE", "NNE", "N", "NNW", "NW", "WNW", 
                "W", "WSW", "SW", "SSW", "S", "ESE", "SE", "ESE"]
    dirticks = dirticks[::2]
    ax.set_xticklabels(dirticks, fontsize=20)
    ax.tick_params(pad=12)
    for tick in ax.get_xticklabels():
        tick.set_path_effects(outline)
        tick.set_color('white')


    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_yticks(ax.get_yticks()[::2])
    for tick in ax.get_yticklabels():
        tick.set_ha('left')
        tick.set_color('white')
        tick.set_fontsize(15)
    ax.set_ylim(0, ax.get_ylim()[1]*1.1)
    ax.set_rlabel_position(0)

    # Background color
    ax.patch.set_facecolor('k')
    ax.patch.set_alpha(.8)
    fig.patch.set_alpha(0.)


    # Making legend elements
    legend_elements = []
    for n in np.flip(range(1, ncountbins+1)):
        legend_elements.append(Patch(facecolor=cmap((n-1)/(ncountbins-1)),
                               edgecolor='k',
                               label=str(int(np.ceil(countbins[n]))) + ' counts')) 

    # Making legend
    leg = plt.legend(handles=legend_elements, 
               bbox_to_anchor=(1,1), 
               bbox_transform=plt.gcf().transFigure,
               fontsize=18,
               frameon=False)

    for text in leg.get_texts():
        text.set_path_effects(outline)
        text.set_color('white')


    # Either save or display the figure
    if save:
        plt.savefig("windrose.png", format='png', facecolor=fig.get_facecolor())
    else:
        plt.show()

    plt.close()





if __name__=="__main__":
    stack_data = gather_data(True)
    rose_data = gather_data(False)

    stack_plot(stack_data, forecast=True, depth=.6, save=True)
    windrose(rose_data, num_days=3, save=True)


