import matplotlib.gridspec as gridspec
import pandas as pd
import matplotlib.ticker as ticker
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator, Locator
from scipy.stats import linregress
import pymc3 as pm
import numpy as np
import matplotlib.gridspec as gridspec

def plot_details(x12, category, i, x, switch, axs):
    """add fits, log, labels"""
    ok = x12[category].notnull() & x12[x].notnull()
    rho, pval = stats.spearmanr(x12[category][ok], x12[x][ok])
    axs[i].annotate(f'{rhotext} = {rho:0.3f} ',
         xy=(0., 0.), xycoords='axes fraction',
         xytext=(.17, 0.88), fontsize=18)
    
    indep = x12[category].values
    axs[i].set_xlim(indep.min() - 0.5, indep.max() + 0.3 * indep.max())
    if (i-2)%3 != 0:
        plt.setp(axs[i].get_yticklabels(), visible=False)
    else:
        axs[i].set_ylabel(lable_dict[x])
        
    if 5-i < 3:
        plt.setp(axs[i].get_xticklabels(), visible=False)
    #else:
        #axs[i].set_xlabel(category)
    axs[i].axvline(x=switch,linestyle="--")
    
def regression(x12,category,i,x,axs,axs2):
    """linear regression"""
    ok = x12.sort_values(by=[category])[category].notnull() & x12.sort_values(by=[category])[x].notnull()
    if category == "Deaths":
        nx = 10**(-1)
    elif category =="Damage":
        nx = 10**(4)
    ny = 10 **(-8)
    X_train = np.log10(x12.sort_values(by=[category])[category][ok].values.astype(float)+nx)
    Y = np.log10(x12.sort_values(by=[category])[x][ok].values+ny)
    # fit
    with pm.Model() as model:
        a0 = pm.Normal("a0",mu=-6,sd=3)
        a1 = pm.Normal("a1",mu=0,sd=1)
        μ = a0+a1*X_train
        sd = pm.HalfCauchy('sd',0.1)
        obs = pm.Normal('obs',mu=μ,sd=sd,observed=Y)
        trace = pm.sample(2000, cores=8,tune=1000)
    pm.traceplot(trace);
    ppc = pm.sample_posterior_predictive(trace, model=model, samples=500) # `y` Should be around ~2.1
    N = 100

    a0s = np.random.choice(trace['a0'], size=N, replace=True)
    a1s = np.random.choice(trace['a1'], size=N, replace=True)
    if category == "Deaths":
        fx1 = np.linspace(-2,4,3000)
    elif category == "Damage":
        fx1 = np.linspace(-1,12,5000)
    models = np.array( [a0 + a1 * fx1 for a0, a1 in zip(a0s, a1s)] )

    mean_model = np.mean(models, axis=0)
    std_model = np.std(models, axis=0)

    for ax in [axs,axs2[i]]:
        ax.plot(10**fx1-nx, 10**mean_model-ny,color=colors[5-i],lw=4)
        if ax == axs2[i]:
            ax.fill_between(10**fx1-nx, 10**(mean_model + std_model)-ny,
                    10**(mean_model - std_model)-ny,
                    color=colors[5-i],lw=1.5,alpha=0.4,edgecolor='grey')

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


def integrated_freq(x12,column='freq'):
    freq = []
    for i,x in enumerate(x12["Data"].values):
        freq.append(0)
        for key,value in x.items():
            if key[0] == 'h':
                freq[i] += np.nansum(value[column]) 

    return freq


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


def cat2(wind,i):
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


font = {'family' : 'serif',
        'weight' : 'normal',
        'size' : 35}

plt.rc('font', **font)








def main():
    colors = ["#ff5c5c","#ee8c39","#f0e74f","#a1d46f","#4fb8a6","#5682b0"]
    label_list = ['Max Frequency','Frequency Sum','Wind']
    list2 = ['Max Frequency','Integrated Frequency','Wind']
    lable_dict = {x:label_list[i] for i,x in enumerate(list2)}
    
    x12 = pd.read_pickle("../data/cleaned_hurricane_updated.pkl") 

    columns = ['freq','freq_noRT']
    for c in columns:
        x12[f'Integrated {c}'] = integrated_freq(x12,c)
    x12['Deaths'] = [int(i) for i in x12["Deaths"].values]
    

    # subset to hurricanes
    x12 = x12[x12['Wind']>73]
    # 1grams
    x = 'Integrated Frequency'

    # 2grams
    #x = 'Integrated freq'

    cat = ["Deaths","Damage"]

    ok = x12[cat[0]].notnull() & x12[x].notnull() & x12[cat[1]].notnull()

    nx = { "Deaths":10**(-1),"Damage":10**(4)}
    ny = 10**(-8)

    X_train = x12
    x1 = np.log10(X_train[cat[0]][ok].values.astype(float)+nx[cat[0]])
    x2 = np.log10(X_train[cat[1]][ok].values.astype(float)+nx[cat[1]])
    Y = np.log10(x12[x][ok].values+ny)

    ########################################
    # Regression 1
    ########################################

    with pm.Model() as model:
        a0 = pm.Normal("a0",mu=-8,sd=3)
        a1 = pm.Normal("Deaths",mu=0,sd=1)
        a2 = pm.Normal('Damage',mu=0,sd=1)
        μ = a0 + a1*x1 + a2*x2
                    
        sd = pm.HalfCauchy('sd',0.1)
        obs = pm.Normal('obs',mu=μ,sd=sd,observed=Y)
        trace = pm.sample(2000, cores=8,tune=1000)

    N=2000
    a0s = np.random.choice(trace['a0'], size=N, replace=True)
    a1s = np.random.choice(trace['Deaths'], size=N, replace=True)
    a2s = np.random.choice(trace['Damage'], size=N, replace=True)

    ais = [a0s,a1s,a2s]
    titles = ["$a_0$", '$a_{deaths}$','$a_{damages}$']
    ylabels = ['log $I_{r_i}', ]
    win = (-1.3,1.5)

    ylims = [None, win, win]
    f, axes = plt.subplots(4, 4, figsize=(7*4, 7*4))
    
    # add boxes
    for i in range(2):
        #outer
        outergs = gridspec.GridSpec(1, 1)
        if i == 0:
            outergs.update(bottom=0.505,left=0.01, 
                       top=0.51,  right=1-0.01)
        if i == 1:
            outergs.update(bottom=0.75,left=0.01, 
                       top=0.755,  right=1-0.01)
        outerax = f.add_subplot(outergs[0])
        outerax.tick_params(axis='both',which='both',bottom=0,left=0,
                            labelbottom=0, labelleft=0)
        outerax.set_facecolor('k')
        outerax.patch.set_alpha(0.3)
        outerax.patch.set_zorder(-100000000)
    
    ax = axes[0,:]
    for i, ai in enumerate(ais):
        parts = ax[i].violinplot(ai,showmeans=True)
        for pc in parts['bodies']:
            pc.set_edgecolor('black')
        ax[i].set_title(titles[i], y=1.05,fontsize=45)
        if ylims[i]:
            ax[i].set_ylim(ylims[i][0],ylims[i][1])
        ax[i].set_xticks([])
    ax[3].axis('off')


    ########################################
    # Regression 2
    ########################################


    with pm.Model() as model:
        a0 = pm.Normal("a0",mu=-8,sd=3)
        a1 = pm.Normal("Deaths",mu=0,sd=1)
        a2 = pm.Normal('Damage',mu=0,sd=1)
        a3 = pm.Normal('Interaction',mu=0,sd=1)
        μ = a0 + a1*x1 + a2*x2 + a3*x1*x2
                    
        sd = pm.HalfCauchy('sd',0.1)
        obs = pm.Normal('obs',mu=μ,sd=sd,observed=Y)
        trace = pm.sample(2000, cores=8,tune=1000)


    N=2000
    a0s = np.random.choice(trace['a0'], size=N, replace=True)
    a1s = np.random.choice(trace["Deaths"], size=N, replace=True)
    a2s = np.random.choice(trace['Damage'], size=N, replace=True)
    a3s = np.random.choice(trace['Interaction'], size=N, replace=True)


    ais = [a0s,a1s,a2s,a3s]
    titles = ["$a_0$", '$a_{deaths}$','$a_{damages}$', '$a_{d,D}$']
    ylabels = ['log $I_{r_i}', ]
    ylims = [None, win, win, win]
    
    ax = axes[1,:]
    for i, ai in enumerate(ais):
        parts = ax[i].violinplot(ai,showmeans=True)
        for pc in parts['bodies']:
            pc.set_edgecolor('black')
        ax[i].set_title(titles[i], y=1.05,fontsize=45)
        if ylims[i]:
            ax[i].set_ylim(ylims[i][0],ylims[i][1])
        ax[i].set_xticks([])
    
    
    ########################################
    # Regression 3
    ########################################
    
    
    x_i = []
    for i in range(2,6):
        x_i.append(np.array([cat2(j,i) for j in X_train['Wind'][ok]]))


    with pm.Model() as model:
        a0 = pm.Normal("a0",mu=-8,sd=3)
        a1 = pm.Normal("Deaths",mu=0,sd=1)
        a2 = pm.Normal('Damage',mu=0,sd=1)
        a3 = pm.Normal('Interaction',mu=0,sd=1)
        
        c = []
        for i in range(1,5):
            c.append(pm.Normal(f'Cat{1+i}',mu=0,sd=1))
            
        μ = a0 + a1*x1 + a2*x2 + a3*x1*x2 + c[0]*x_i[0] + c[1]*x_i[1] + c[2]*x_i[2] + c[3]*x_i[3]
                    
        sd = pm.HalfCauchy('sd',0.1)
        obs = pm.Normal('obs',mu=μ,sd=sd,observed=Y)
        trace = pm.sample(2000, cores=8,tune=1000)


    N = 2000
    a_names = ["a0", "Deaths","Damage","Interaction","Cat2","Cat3","Cat4","Cat5"]
    ais = [np.random.choice(trace[f'{i}'], size=N, replace=True) for i in a_names]

    titles = ["$a_0$", '$a_{deaths}$','$a_{damages}$', '$a_{d,D}$']
    titles.extend(['$a_{cat 2}$','$a_{cat 3}$','$a_{cat 4}$','$a_{cat 5}$'])
    ylabels = ['log $I_{r_i}', ]
    win2 = (-1,2.5)
    ylims = [None, win, win,win,win2,win2,win2,win2]
    colors = ['#E02026','#E45825','#E97D25','#EF9F22','#F6C21C','#FEE606']
    
    ax = axes[2:,:].ravel() 
    for i, ai in enumerate(ais):
        if i > 3:
            for k in range(2):
                parts = ax[i].violinplot(ai,showmeans=True)
                for pc in parts['bodies']:
                    pc.set_facecolor(colors[7-i])
                    pc.set_edgecolor('black')

        else:
            parts = ax[i].violinplot(ai,showmeans=True)
            for pc in parts['bodies']:
                pc.set_edgecolor('black')
        ax[i].set_title(titles[i], y=1.05,fontsize=45)
        if ylims[i]:
            ax[i].set_ylim(ylims[i][0],ylims[i][1])
        ax[i].set_xticks([])
    
    letters = "ABCDEFGHIJKLMNOP"
    for i, ax in enumerate(axes.ravel()):
        if i < 3:
            ax.text(0.09, 0.82, letters[i], transform=ax.transAxes,
                fontsize=45,
                bbox=dict(boxstyle="round",
                   ec=(0., 0., 0.),
                   fc=(1., 1., 1.),
                   ))
        elif i > 3:
            ax.text(0.09, 0.82, letters[i-1], transform=ax.transAxes,
                fontsize=45,
                bbox=dict(boxstyle="round",
                   ec=(0., 0., 0.),
                   fc=(1., 1., 1.),
                   ))
    plt.tight_layout()

    plt.savefig("../figures/regression1.png")
    plt.savefig("../figures/regression1.pdf")
                
if __name__ == "__main__":
    main()
