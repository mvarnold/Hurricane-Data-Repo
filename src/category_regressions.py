import matplotlib.gridspec as gridspec
import pandas as pd
import matplotlib.ticker as ticker
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator, Locator
from scipy.stats import linregress
import pymc3 as pm
import seaborn as sns
import numpy as np
import pymc3 as pm
font = {'family' : 'serif',
        'weight' : 'normal',
        'size' : 30}

plt.rc('font', **font)
np.random.seed(1523)

def cat1(wind):
    if wind > 73:
        if wind < 95:
            return 1
        elif wind < 110:
            return 2
        elif wind < 129:
            return 3
        elif wind < 156:
            return 4
        elif wind > 156:
            return 5
    else:
        return 0

def cat2(wind, i):
    cat = cat1(wind)
    if cat == i:
        return 1
    else:
        return 0

def labels(x):
    if x == 0:
        return "Tropical Storm"
    else:
        return f"Category {x}"

class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """
    def __init__(self, linthresh, nints=10):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically. nints gives the number of
        intervals that will be bounded by the minor ticks.
        """
        self.linthresh = linthresh
        self.nintervals = nints

    def __call__(self):
        # Return the locations of the ticks
        majorlocs = self.axis.get_majorticklocs()

        if len(majorlocs) == 1:
            return self.raise_if_exceeds(np.array([]))

        # add temporary major tick locs at either end of the current range
        # to fill in minor tick gaps
        dmlower = majorlocs[1] - majorlocs[0]    # major tick difference at lower end
        dmupper = majorlocs[-1] - majorlocs[-2]  # major tick difference at upper end

        # add temporary major tick location at the lower end
        if majorlocs[0] != 0. and ((majorlocs[0] != self.linthresh and dmlower > self.linthresh) or (dmlower == self.linthresh and majorlocs[0] < 0)):
            majorlocs = np.insert(majorlocs, 0, majorlocs[0]*10.)
        else:
            majorlocs = np.insert(majorlocs, 0, majorlocs[0]-self.linthresh)

        # add temporary major tick location at the upper end
        if majorlocs[-1] != 0. and ((np.abs(majorlocs[-1]) != self.linthresh and dmupper > self.linthresh) or (dmupper == self.linthresh and majorlocs[-1] > 0)):
            majorlocs = np.append(majorlocs, majorlocs[-1]*10.)
        else:
            majorlocs = np.append(majorlocs, majorlocs[-1]+self.linthresh)

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i-1]
            if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
                ndivs = self.nintervals
            else:
                ndivs = self.nintervals - 1.

            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                          '%s type.' % type(self))



def plot_details(x12, category, i, x, switch, axs):
    """add fits, log, labels"""
    ok = x12[category].notnull() & x12[x].notnull()
    rho, pval = stats.spearmanr(x12[category][ok], x12[x][ok])
    axs[i].annotate(f'{rhotext} = {rho:0.3f} ',
                    xy=(0., 0.), xycoords='axes fraction',
                    xytext=(.17, 0.88), fontsize=24)
    # axs[i].set_xscale('log')
    # axs[i].set_yscale('log')
    #
    #
    indep = x12[category].values
    axs[i].set_xlim(indep.min() - 0.5, indep.max() + 0.3 * indep.max())
    if (i - 2) % 3 != 0:
        plt.setp(axs[i].get_yticklabels(), visible=False)
    else:
        if (i - 2) % 6 == 0:
            axs[i].set_ylabel(lable_dict[x], y=1)

    if 5 - i < 3:
        plt.setp(axs[i].get_xticklabels(), visible=False)
    # else:
    # axs[i].set_xlabel(category)
    axs[i].axvline(x=switch, linestyle="--")


def regression(x12, category, i, x, axs, axs2):
    """linear regression"""
    ok = x12.sort_values(by=[category])[category].notnull() & x12.sort_values(by=[category])[x].notnull()
    if category == "Deaths":
        nx = 10 ** (-1)
    elif category == "Damage":
        nx = 10 ** (4)
    ny = 10 ** (-8)
    X_train = np.log10(x12.sort_values(by=[category])[category][ok].values.astype(float) + nx)
    Y = np.log10(x12.sort_values(by=[category])[x][ok].values + ny)
    # fit
    with pm.Model() as model:
        tau = pm.Gamma('tau', 3.0, 1.0)
        a0 = pm.Normal("a0", mu=-8, tau=tau)
        a1 = pm.Normal("a1", mu=0.0, sd=1)

        μ = a0 + a1 * X_train
        sd = pm.HalfCauchy('sd', 0.2)
        obs = pm.Normal('obs', mu=μ, sd=sd, observed=Y)
        trace = pm.sample(2000, cores=8, tune=1000)
    pm.traceplot(trace)
    plt.suptitle(f"{category} Cat {i}")
    fig = plt.gcf()  # to get the current figure...
    fig.savefig(f"traceplot_{category}_{i}.png")
    ppc = pm.sample_posterior_predictive(trace, model=model, samples=500)  # `y` Should be around ~2.1
    N = 100

    a0s = np.random.choice(trace['a0'], size=N, replace=True)
    a1s = np.random.choice(trace['a1'], size=N, replace=True)
    if category == "Deaths":
        fx1 = np.linspace(-2, 4, 3000)
    elif category == "Damage":
        fx1 = np.linspace(-1, 12, 5000)
    models = np.array([a0 + a1 * fx1 for a0, a1 in zip(a0s, a1s)])

    mean_model = np.mean(models, axis=0)
    std_model = np.std(models, axis=0)

    for ax in [axs, axs2[i]]:
        ax.plot(10 ** fx1 - nx, 10 ** mean_model - ny, color=colors[5 - i], lw=4)
        if ax == axs2[i]:
            ax.fill_between(10 ** fx1 - nx, 10 ** (mean_model + std_model) - ny,
                            10 ** (mean_model - std_model) - ny,
                            color=colors[5 - i], lw=1.5, alpha=0.4, edgecolor='grey')
    print(f"Cat {i}: {category}")
    print(pm.summary(trace).round(2))
    return category, i, a1s

x12 = pd.read_pickle("../data/cleaned_hurricane_updated.pkl")
x12['Deaths'] = [int(i) for i in x12["Deaths"].values]
label_list = ['Max Usage Rate','Usage Rate Sum','Wind']
list2 = ['Max Frequency','Integrated Frequency','Wind']
lable_dict = {x:label_list[i] for i,x in enumerate(list2)}

colors = ['#E02026', '#E45825', '#E97D25', '#EF9F22', '#F6C21C', '#FEE606']

offset = -0.1
f = plt.figure(figsize=(25, 18))
gs1 = gridspec.GridSpec(8, 9)
gs1.update(left=0.05, right=0.47, top=.95, bottom=0.48 - offset, wspace=0.5, hspace=0.5)
ax11 = plt.subplot(gs1[:4, :3])
ax12 = plt.subplot(gs1[:4, 3:6])
ax13 = plt.subplot(gs1[:4, 6:])
ax14 = plt.subplot(gs1[4:, :3])
ax15 = plt.subplot(gs1[4:, 3:6])
ax16 = plt.subplot(gs1[4:, 6:])

gs2 = gridspec.GridSpec(8, 9)
gs2.update(left=0.55, right=0.98, top=.95, bottom=0.48 - offset, hspace=0.5, wspace=0.5)
ax21 = plt.subplot(gs2[:4, :3])
ax22 = plt.subplot(gs2[:4, 3:6])
ax23 = plt.subplot(gs2[:4, 6:])
ax24 = plt.subplot(gs2[4:, :3])
ax25 = plt.subplot(gs2[4:, 3:6])
ax26 = plt.subplot(gs2[4:, 6:])

gs3 = gridspec.GridSpec(9, 6)
gs3.update(left=0.05, right=0.47, top=.44 - offset, bottom=0.05)
ax7 = plt.subplot(gs3[:, :])

gs4 = gridspec.GridSpec(9, 6)
gs4.update(left=0.55, right=0.98, top=.44 - offset, bottom=0.05)
ax8 = plt.subplot(gs4[:, :])

ax = [ax7, ax8]
sub_ax1 = [ax11, ax12, ax13, ax14, ax15, ax16][::-1]
sub_ax2 = [ax21, ax22, ax23, ax24, ax25, ax26][::-1]

handles = []
rhotext = r'$\rho_s$'
data_list = []

for i in range(5, -1, -1):

    x11 = x12[x12.apply(lambda x: cat1(x['Wind']) == i, axis=1)]

    for j, x in enumerate(['Integrated Frequency']):

        ax[j * 2].plot(x11['Deaths'].values, x11[x].values, '.', color='grey')
        ax[j * 2].plot(x11['Deaths'].values, x11[x].values, 'o', ms=22, alpha=0.5, color=colors[5 - i])

        sub_ax1[i].plot(x11['Deaths'].values, x11[x].values, '.', color='grey')
        sub_ax1[i].plot(x11['Deaths'].values, x11[x].values, 'o', ms=15, alpha=0.5, color=colors[5 - i])

        plot_details(x11, "Deaths", i, x, 1.1, sub_ax1)

        data_list.append(regression(x11, "Deaths", i, x, ax[j * 2], sub_ax1))

        if len(handles) < 6 and j == 0:
            handles.append(Polygon([(0, 0), (10, 0), (0, -10)], color=colors[5 - i],
                                   label=labels(i)))

        ok = x11['Damage'].notnull() & x11[x].notnull()
        # rho, pval = stats.spearmanr(x11['Damage'][ok],x11[x][ok])
        ax[j * 2 + 1].plot(x11['Damage'].values, x11[x].values, '.', color='grey')
        ax[j * 2 + 1].plot(x11['Damage'].values, x11[x].values, 'o', ms=22, alpha=0.5, color=colors[5 - i])

        sub_ax2[i].plot(x11['Damage'].values, x11[x].values, '.', color='grey')
        sub_ax2[i].plot(x11['Damage'].values, x11[x].values, 'o', ms=15, alpha=0.5, color=colors[5 - i])
        plot_details(x11, "Damage", i, x, 25000, sub_ax2)

        data_list.append(regression(x11, "Damage", i, x, ax[j * 2 + 1], sub_ax2))

for j, x in enumerate(['Integrated Frequency']):
    ok = x12['Deaths'].notnull() & x12[x].notnull()
    rho, pval = stats.spearmanr(x12['Deaths'][ok], x12[x][ok])
    ax[j * 2].annotate(f'{rhotext} = {rho:0.3f} ',
                       xy=(0., 0.), xycoords='axes fraction',
                       xytext=(.2, 0.88), fontsize=28)

    ax[j * 2].set_xscale('symlog', linthreshx=1.1)
    ax[j * 2].set_yscale('symlog', linthreshy=10 ** (-9))
    for i in sub_ax1:
        i.set_xscale(ax[j * 2].get_xscale(), linthreshx=1.1)
        i.set_yscale('symlog', linthreshy=10 ** (-9))
        i.set_xlim((-0.5, 3974.1))
        i.set_ylim((-3.236081193455171e-10, 0.03232762235178018))
        i.set_yticks(i.get_yticks()[::2])
        i.tick_params(axis='both', which='major', labelsize=24)

    ax[j * 2].set_ylabel(lable_dict[x])
    ax[j * 2].set_xlabel("Deaths")
    indep = x12["Deaths"].values
    ax[j * 2].set_xlim(indep.min() - 0.5, indep.max() + 0.3 * indep.max())
    ax[j * 2].axvline(x=1.1, linestyle="--")

    ok = x12['Damage'].notnull() & x12[x].notnull()
    rho, pval = stats.spearmanr(x12['Damage'][ok], x12[x][ok])
    ax[j * 2 + 1].annotate(f'{rhotext}= {rho:0.3f} ',
                           xy=(0., 0.), xycoords='axes fraction',
                           xytext=(.2, 0.88), fontsize=28)

    ax[j * 2 + 1].set_xscale('symlog', linthreshx=28000)
    ax[j * 2 + 1].set_yscale('symlog', linthreshy=10 ** (-9))
    for i in sub_ax2:
        i.set_xlim((-9778.687496898952, 305459170209.8102))
        i.set_ylim((-3.236081193455171e-10, 0.03232762235178018))
        i.set_xscale('symlog', linthreshx=28000)
        i.set_yscale('symlog', linthreshy=10 ** (-9))
        i.set_xticks(i.get_xticks()[::2])
        i.set_yticks(i.get_yticks()[::2])
        i.tick_params(axis='both', which='major', labelsize=24)

        # xloc = plt.MaxNLocator(50)
        # i.xaxis.set_major_locator(xloc)

    ax[j * 2 + 1].set_ylabel(lable_dict[x])
    ax[j * 2 + 1].set_xlabel("Damage")
    indep = x12["Damage"].values
    # ax[j*2+1].set_xlim(indep.min()-5000.5, indep.max()+0.7*indep.max())
    ax[j * 2 + 1].axvline(x=28000, linestyle="--")
    ax[j * 2 + 1].get_xscale()
    ax[j * 2 + 1].set_xlim((-9778.687496898952, 305459170209.8102))
    ax[j * 2 + 1].set_ylim((-3.236081193455171e-10, 0.03232762235178018))
    ax[j * 2].set_xlim((-0.5, 3974.1))
    ax[j * 2].set_ylim((-3.236081193455171e-10, 0.03232762235178018))

ax[1].legend(handles=handles, loc="upper left", bbox_to_anchor=(-1.2, -0.1), prop={'size': 27}, ncol=6)
ax[1].xaxis.set_minor_locator(MinorSymLogLocator(0.5))
ax[0].xaxis.set_minor_locator(MinorSymLogLocator(25000))
f.subplots_adjust(left=0.125, bottom=0.1)
f.savefig("../figures/catn_regression.png", bbox_inches='tight')
f.savefig("../figures/catn_regression.pdf", bbox_inches='tight')