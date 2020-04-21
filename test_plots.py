from make_data import *
from make_plots import *

data = placeholder_data()
stack_plot(data, forecast=True, depth=.6, save=False)
windrose(data, save=False)





