import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rc
    

dataset_ordering = ('mediamill', 'Yelp', 'corel5k', 'bibtex', 'enron', 'medical', 'birds', 'emotions', 'CAL500')

def set_mpl_style():
    plt.style.use(['seaborn-notebook', 'seaborn-darkgrid'])
    mpl.rcParams['axes.facecolor'] = 'white'
    mpl.rcParams['grid.color'] = 'lightgray'
    plt.rcParams["font.family"] = "serif"

    params = {'legend.fontsize': 'x-large',
              'legend.title_fontsize': 'x-large',
              # 'figure.figsize': (15, 5),
             'axes.labelsize': 'x-large',
             'axes.titlesize':'x-large',
             'xtick.labelsize':'x-large',
             'ytick.labelsize':'x-large'}
    plt.rcParams.update(params)

    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('text', usetex=True)

