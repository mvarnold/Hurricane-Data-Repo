import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os
import sys
import conda
print('hello')
conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
import matplotlib.dates as mdates

import pandas as pd
import datetime

# to do: switch to new database
sys.path.insert(0, '/home/michael/.passwords')
from mongo_password import x as pwd
sys.path.insert(0, '/home/michael/ngram_query')
from mongo_query import Query

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
from scipy.interpolate import interp1d

colors = ["#2078B5", "#FF7F0F", "#2CA12C", "#D72827", "#9467BE", "#8C574B",
            "#E478C2", "#808080", "#BCBE20", "#17BED0", "#AEC8E9", "#FFBC79", 
            "#98E08B", "#FF9896", "#C6B1D6", "#C59D94", "#F8B7D3", "#C8C8C8", 
           "#DCDC8E", "#9EDAE6"]
color_len = len(colors)


font = {'family' : 'serif',
        'weight' : 'normal',
        'size' : 18}

plt.rc('font', **font)

def plot_timeseries(word_list,begin_date,end_date, axs):
    data = {}
    dlist = []
    ylist = []
    for word_i in word_list:
        # todo: switch to new database
        query =  Query('mvarnold', pwd, '1grams', 'en' )
        data_i = query.query_insensitive_timeseries({'word' : word_i,'time': {'$gte': begin_date,'$lte':end_date}},word_i)
        try:
            count_i, rank_i, freq_i = data_i.values.T 
        except:
            return
        if freq_i.size > 0:
            data[word_i] = {}
            data[word_i]['count'] = count_i 
            data[word_i]['rank'] = rank_i
            data[word_i]['freq'] = freq_i 
            data[word_i]['date'] = data_i.index
        else:
            word_list.remove(word_i)

    n_words = len(word_list)
    for i, word_i in enumerate(word_list):
        axs[1].plot(data[word_i]['date'], data[word_i]['freq'], 's-',ms=2, color=colors[i%color_len], alpha=0.8, label=word_i)
        axs[2].semilogy(data[word_i]['date'], data[word_i]['freq'], 's-',ms=2, color=colors[i%color_len], alpha=0.8, label=word_i)
        axs[1].fill_between(data[word_i]['date'], y1=data[word_i]['freq'], alpha=0.2, facecolor=colors[i%color_len], interpolate=False)
        axs[2].fill_between(data[word_i]['date'], y1=data[word_i]['freq'], alpha=0.2, facecolor=colors[i%color_len], interpolate=False)
        
        maxx,maxy = data[word_i]['date'][np.argmax(data[word_i]['freq'])], np.max(data[word_i]['freq'])
        axs[1].plot(maxx, maxy, marker="*", color=colors[i%color_len], alpha=0.8, ms=10)
        axs[2].semilogy(maxx, maxy, "*", color=colors[i%color_len], alpha=0.8, ms=10)        
        #draw lines for beginning, and max day, and last map day.
        dlist.append(maxx)
        ylist.append(maxy)
    axs[1].set_ylabel('Frequency')
    axs[2].set_ylabel('Frequency')
    axs[2].set_xlabel('Date')
    return dlist,ylist

def purpendicular(x,y):
    x,y = -y,x

    norm = np.sqrt(x**2+y**2)
    x = x / norm
    y = y / norm
    return x,y

def cat1(wind):
        if wind > 64:
            if wind < 82:
                return 1
            elif wind < 95:
                return 2
            elif wind < 112:
                return 3
            elif wind < 136:
                return 4
            return 5
        else:
            return 0


def unit_perp(x,y,oldx,oldy,scale,axes,smooth=0.5,ooldx=None,ooldy=None):
    x1,y1 = purpendicular(oldx-x,oldy-y)
    if ooldx:
        x2,y2 = purpendicular(ooldx-oldx,ooldy-oldy)
        x1,y1 = (x1+x2*smooth, y1+y2*smooth)
    sc = 3*10e+3
    norm = np.sqrt(x1**2+y1**2)
    x1 = x1 / norm
    y1 = y1 / norm
    # uncomment for unit purpendicular lines
    #axes[0].plot(np.array([x+x1*sc,x-x1*sc]),np.array([y+y1*sc,y-y1*sc]),lw=2, color='k',alpha=0.3)
    return x+x1*scale,x-x1*scale,y+y1*scale,y-y1*scale

def get_scale(name,date):
    word_i = '#hurricane'+name.lower()
    date = datetime.datetime(date.year,date.month, date.day)
    query =  Query('mvarnold', pwd, '1grams', 'en' )
    data_i = query.query_insensitive_timeseries({'word' : word_i, 'time': date},word_i)
    try:
        count_i, rank_i, freq_i = data_i.values.T 
    
        return freq_i
    except:
        return 0

def chaikins_corner_cutting(coords, refinements=5):
    coords = np.array(coords)

    for _ in range(refinements):
        L = coords.repeat(2, axis=0)
        R = np.empty_like(L)
        R[0] = L[0]
        R[2::2] = L[1:-1:2]
        R[1:-1:2] = L[2::2]
        R[-1] = L[-1]
        coords = L * 0.75 + R * 0.25

    return coords


def moving_average(a, n=3) :
    
    """a = np.ravel(a)
    weights = np.array([(i+1)*(n-i)/n for i in range(n)])
    weights = weights/np.sum(weights)
    sma = np.convolve(a, weights, 'same')
    return sma"""
    ret = np.cumsum(a, dtype=float)
    weights = np.array([(i+1)*(n-i)/n for i in range(n)])
    weights = weights/np.sum(weights)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n



def polygon_cooridinates(time_dict,name,year,smooth=4,axes=None):
    x,y,t = zip(*time_dict[name])
    #plt.plot(x,y)
    x_list = []
    y_list = []
    x1_list = []
    y1_list = []
    old_x,old_y = 0,0
    for i1,i in enumerate(zip(x,y)):
        #x,y = i
        if i1 == len(x)-1:
            break
        if i1 > 2:
            x1,x2,y1,y2 = unit_perp(x[1+i1],y[1+i1],x[i1],y[i1],get_scale(name,t[i1])*1.1*10e+8,axes, 1, x[i1-1],y[i1-1])
        else:
            x1,x2,y1,y2 = unit_perp(x[1+i1],y[1+i1],x[i1],y[i1],get_scale(name,t[i1])*1.1*10e+8,axes)
        x_list.append(x1)
        x1_list.append(x2)
        y_list.append(y1)
        y1_list.append(y2)
        old_x,old_y = x,y

    # Smoothing
    x_list = moving_average(x_list,smooth)
    y_list = moving_average(y_list,smooth)
    x1_list = moving_average(x1_list,smooth)
    y1_list = moving_average(y1_list,smooth)


    ylist = np.concatenate((np.array(y_list),np.array(y1_list[::-1])),axis=None)
    xlist = np.concatenate((np.array(x_list),np.array(x1_list[::-1])),axis=None)
    
    #return chaikins_corner_cutting(xlist[::2]),chaikins_corner_cutting(ylist[::2])
    t = np.arange(len(xlist))
    ti = np.linspace(0, t.max(), 10 * t.size)

    xi = interp1d(t, xlist, kind='cubic')(ti)
    yi = interp1d(t, ylist, kind='cubic')(ti)
    #hull = ConvexHull(np.array(
    return xi,yi


def map_plots(df, year, wind):
    """ Make those oh-so-beautiful maps"""
    print(f"Map plots for {year}: wind speed above {wind}")
    fig = plt.figure(figsize=(16,17))

    gs0 = gridspec.GridSpec(16, 1, figure=fig,hspace=2.5)

    gs00 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[:-6],hspace=2.)

    ax1 = plt.Subplot(fig, gs00[:, :])
    fig.add_subplot(ax1)


    gs01 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[-6:],hspace=0.0)

    ax2 = plt.Subplot(fig, gs01[0, :])
    fig.add_subplot(ax2)
    ax3 = plt.Subplot(fig, gs01[1, :],sharex=ax2)
    fig.add_subplot(ax3)
    axes = [ax1,ax2,ax3]

    # Lambert Conformal Conic map.
    m = Basemap(llcrnrlon=-100.,llcrnrlat=0.,urcrnrlon=-30.,urcrnrlat=45.,
                projection='lcc',lat_1=20.,lat_2=42.,lon_0=-60.,
                resolution ='l',area_thresh=1000. , ax=axes[0])
    m.drawcoastlines()
    m.drawparallels(np.arange(10,70,20),labels=[1,1,0,0])
    names = df[(df['Season']==str(year)) &(df['Wind']>wind)]['Name'].values
    data = df[(df['Season']==str(year)) &(df['Wind']>wind)]
    time_dict = {i:[] for i in names}
    handles = []
    
    for index, row in enumerate(data["Positions"].values.T):
        lats = row[:,1]
        lons = row[:,2]
        lons, lats = m(lons, lats)
        key = data["Name"].values[index]
        handles.append(Polygon([(0,0),(10,0),(0,-10)],color=colors[index%color_len],
                               label='Hurricane %s'%(key)))
        m.plot(lons,lats,'s-',ms=2,markevery=8,linewidth=1,color=colors[index%color_len])
        
        print(key)
        # wrong dates going in to time_dict?
        for x in zip(lons, lats, row[:,0]):
            time_dict[key].append(list(x))
    #print(time_dict)
    
    for index, key in enumerate(data["Name"].values):

    # break for adding twitter data
        xlist,ylist = polygon_cooridinates(time_dict,key,year,4,axes)
        poly = Polygon(np.column_stack([xlist,ylist]),alpha=0.5,color=colors[index%color_len],label = "Hurricane %s"%(key))
        ax1.add_patch(poly)

    print(handles)
    lgd = axes[0].legend(loc = "upper left", bbox_to_anchor=(0.0,0.00),ncol=len(handles))
    
    try:
        dlist,ylist = plot_timeseries(['#hurricane'+i.lower() for i in data["Name"].values],
                    datetime.datetime(year,1,1),
                    datetime.datetime(year,12,1), axes)
    except:
        print("No hurricanes")
        return

    for index, row in enumerate(data["Positions"].values.T):
        lats = row[:,1]
        lons = row[:,2]
        dates = row[:,0]
        lons, lats = m(lons, lats)
        key = data["Name"].values[index]
        date = datetime.datetime.combine(dlist[index],datetime.time(12,0))
        m.plot(lons[dates == date],lats[dates==date],'*',ms=10,color=colors[index%color_len])

    locator = mdates.WeekdayLocator(byweekday=MO, interval=2)
    fmt = mdates.DateFormatter('%b-%d')
    ax3.xaxis.set_major_locator(locator)
    ax3.xaxis.set_major_formatter(fmt)
    ax2.grid(which='major', axis='x',lw=1.5,alpha=0.5,linestyle = '--')
    ax3.grid(which='major', axis='x',lw=1.5,alpha=0.5,linestyle = '--')

    # add labels
    ax1.text(0.1, 0.9, "A")
    ax2.text(0.1, 0.9, "B")
    ax3.text(0.1, 0.9, "C")

    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.savefig(f"../figures/{str(year)}.pdf",bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


def main():
    df = pd.read_pickle("../data/cleaned_hurricane_updated.pkl")
    #for i in range(2009,2019,1):
    #    map_plots(df,i,110)
    map_plots(df,2017,125)

if __name__=='__main__':
    main()
