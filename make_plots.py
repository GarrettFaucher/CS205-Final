import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch
import random
import pandas as pd
from make_data import linearmap


BACKGROUND_COLOR = (.68,.45,.33, .5)
ROSE_CMAP = 'cividis'

# Functional form that we attempt to forecast temp and pressure with
def fitfunc1(x, a0, a1, a2, a3, b0, b1, b2):
    return a0 + a1*x + a2*x**2 + a3*x**3 + b0*np.sin(x*b1 + b2)

# Functional form that we attempt to forecast wind speed with
def fitfunc2(x, a0, a1, a2, a3):
    return a0 + a1*x + a2*x**2 + a3*x**3

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

    # Plotting temperature
    ax[0].plot(data['time'], data['temp'], **plot_kwargs)
    ax[0].set_ylabel(r"Temperature ($^\circ$C)", fontsize=14)
    ax[0].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    # Plotting pressure
    ax[1].plot(data['time'], data['pressure'], **plot_kwargs)
    ax[1].set_ylabel("Pressure (atm)", fontsize=14)
    ax[1].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2f}'))

    # Plotting wind speed
    ax[2].plot(data['time'], data['windspeed'], **plot_kwargs)
    ax[2].set_ylabel("Wind speed (mph)", fontsize=14)
    ax[2].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2f}'))

    # Setting time ticks and labelling x-axis
    x_ticks = [linearmap([0, 10],[min_time,max_time],i) for i in range(0,10)]
    ax[2].set_xticks(x_ticks)
    ax[2].set_xlabel("Time (sol)", fontsize=16)

    # kwargs to pass to grid lines
    tickline_kwargs = dict(color='k',
                           linestyle='--',
                           linewidth=.5,
                           alpha=.6)
    # Vertical grid lines
    for x in x_ticks:
        ax[0].axvline(x, **tickline_kwargs)
        ax[1].axvline(x, **tickline_kwargs)
        ax[2].axvline(x, **tickline_kwargs)

    s = .2 # Scale to set margin between data points and borders of graphs
    for col in data.columns[1:-1]:
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
        fit_data = data[:][3*N//4:]
        t = np.linspace(data['time'].iloc[-1], max_time, 100)

        # Do forecast on temp
        temp_params,cov = spo.curve_fit(fitfunc1, data['time'], data['temp'])
        temp_fit = fitfunc1(t, *temp_params)
        scale = .1*max(data['temp'] - min(data['temp']))
        ax[0].fill_between(t, temp_fit - scale*(t[0] - t), temp_fit + scale*(t[0] - t),
                           facecolor='r',
                           alpha=.3)
        ax[0].plot(t, temp_fit, 'r--')

        # Do forecast on pressure
        pressure_params,cov = spo.curve_fit(fitfunc1, data['time'], data['pressure'])
        pressure_fit = fitfunc1(t, *pressure_params)
        scale = .1*max(data['pressure'] - min(data['pressure']))
        ax[1].fill_between(t, pressure_fit - scale*(t[0] - t), pressure_fit + scale*(t[0] - t),
                           facecolor='r',
                           alpha=.3)
        ax[1].plot(t, pressure_fit, 'r--')

        # Do forecast on pressure
        speed_params,cov = spo.curve_fit(fitfunc2, data['time'][3*N//4:], data['windspeed'][3*N//4:])
        speed_fit = fitfunc2(t, *speed_params)
        scale = .1*max(data['windspeed'] - min(data['windspeed']))
        ax[2].fill_between(t, speed_fit - scale*(t[0] - t), speed_fit + scale*(t[0] - t),
                           facecolor='r',
                           alpha=.3)
        ax[2].plot(t, speed_fit, 'r--')



    fig.align_ylabels()
    fig.patch.set_alpha(0.)

    # Either save or display the figure
    if save:
        plt.savefig("stackplot.jpeg", format='jpeg', transparent='True')
    else:
        plt.show()

    plt.close()

# Make wind rose plot
def windrose(data, save=False):
    fig, ax = plt.subplots(nrows=1, ncols=1,
                           figsize=(10,10),
                           subplot_kw=dict(projection='polar'))

    # Loading data (and converting winddirection to radians)
    wdir = np.pi/180.*data['winddirection']
    wspeed = data['windspeed']

    # Setting bin parameters
    N = len(wspeed)
    ndirbins = 16
    width = 2*np.pi/ndirbins
    dirbins = np.arange(0, 2*np.pi, width)

    nspeedbins = 4
    speedbins = np.linspace(0, max(wspeed), nspeedbins+1)
    cmap = plt.cm.get_cmap(ROSE_CMAP)

    # Iterate backwards through speedbins
    for n in np.flip(range(1, nspeedbins+1)):
        speed = speedbins[n]

        # Binning data
        counts, bins = np.histogram(wdir[wspeed < speed], bins=dirbins)
        radii = 100.*counts/N

        # Plotting data
        ax.bar(bins[:-1], radii, align='edge', color=cmap((n-1)/(nspeedbins-1)), width=width)


    # Setting tick properties
    ax.set_xticks(dirbins)
    dirticks = ["E", "ENE", "NE", "NNE", "N", "NNW", "NW", "WNW", 
                "W", "WSW", "SW", "SSW", "S", "ESE", "SE", "ESE"]
    ax.set_xticklabels(dirticks, fontsize=20)
    ax.tick_params(pad=12)

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


    legend_elements = []
    for n in np.flip(range(1, nspeedbins+1)):
        legend_elements.append(Patch(facecolor=cmap((n-1)/(nspeedbins-1)),
                               edgecolor='k',
                               label='<{:.1f} mph'.format(speedbins[n])))
    plt.legend(handles=legend_elements, 
               bbox_to_anchor=(1,1), 
               bbox_transform=plt.gcf().transFigure,
               fontsize=18,
               frameon=False)

    # Either save or display the figure
    if save:
        plt.savefig("windrose.jpeg", format='jpeg')
    else:
        plt.show()

    plt.close()




