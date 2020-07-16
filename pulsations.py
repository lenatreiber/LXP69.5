import numpy as np
from uncertainties import ufloat, unumpy
from uncertainties.umath import *
from astropy.io import fits
from astropy.table import Table
import pandas as pd
import matplotlib.pyplot as plt
from astropy.stats import LombScargle
from scipy import signal
from stingray.pulse.search import epoch_folding_search, z_n_search
from matplotlib.gridspec import GridSpec
from stingray.pulse.search import phaseogram, plot_phaseogram, plot_profile
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

def readevt(file,ind=1):
    '''
    Reads evt file and returns astropy Table
    '''
    hdu_list = fits.open(file, memmap=True)
    return Table(hdu_list[ind].data)

def phasehist(evt_data,pd,figsize=(8,6),color='slateblue',bins=32):
    '''
    Plots step histogram of phase
    
    Required Inputs:
    ----------------
    evt_data: astropy table of events
    pd: period for phase-folding
    
    Optional Inputs:
    ----------------
    figsize: tuple of matplotlib figuresize
             default (8,6)
    color: matplotlib color of hist
           default slateblue
    bins: int of bins to use in hist
          default 32 (16 phase bins)
          
    Outputs:
    --------
    a: histogram: array where 0 index array is counts in each bin;
                  index 1 array is bin boundaries
    mids: list of phase bin centers
    '''
    plt.figure(figsize=figsize)
    ph = evt_data['TIME']%pd 
    ph_1 = ph/pd #divides phase by period so that x-axis goes to 1, not pd
    ph_2 = ph_1+1 
    ph_3 = list(ph_1)+list(ph_2) #duplicates and concatenates 
    a = plt.hist(ph_3,bins=bins,color='slateblue',histtype='step',label='all')
    plt.xlim(0,2)
    plt.xlabel('Phase',fontsize=14)
    plt.ylabel('Counts',fontsize=14)
    #constructs list of middle of each phase bin
    mids = []
    for t in range(1,len(a[1])):
        mids.append(np.mean((a[1][t],a[1][t-1])))
    #adds error as sqrt of counts
    plt.errorbar(mids,a[0],yerr=np.sqrt(a[0]),linestyle='none',color=color)
    #returns histogram (counts and bin boundary arrays), and array of middles of phase bins
    return a,mids

def phasehist_sh(ed,pd,ens=['0.3-10 keV','0.3-1.5 keV','1.5-10 keV'],colors=['slateblue','palevioletred','navy','cornflowerblue'],figsize=(8,6),bins=32):
    '''
    Same as phasehist, but for a list of astropy tables
    
    Inputs:
    -------
    ed: list of evt_data (astropy tables) to be plotted
    pd: period to use to phase-fold
    
    Optional Inputs: 
    ----------------
    ens: list of energy ranges corresponding (in order) to ed
    colors: list of colors to use in hist, same length and order as 
    figsize
    bins: total number of bins
          default is 32 = 16 phase bins
    Outputs:
    --------
    same as phasehist but each as a list
    
    To Do:
    ------
    Check for errors in lists and deal with some automatically (ex: no colors)
    '''
    plt.figure(figsize=figsize)
    hists = []
    for e in range(len(ed)):
        ph = ed[e]['TIME']%pd
        ph_1 = ph/pd
        ph_2 = ph_1+1
        ph_3 = list(ph_1)+list(ph_2)
        hist = plt.hist(ph_3,bins=bins,color=colors[e],histtype='step',label=ens[e])
        hists.append(hist)
        
        if e == 0:
            mids = []
            for t in range(1,len(hist[1])):
                mids.append(np.mean((hist[1][t],hist[1][t-1])))
        plt.errorbar(mids,hist[0],yerr=np.sqrt(hist[0]),linestyle='none',color=colors[e])

    plt.xlim(0,2)
    plt.xlabel('Phase',fontsize=14)
    plt.ylabel('Counts',fontsize=14)
    plt.legend()
    return hists,mids
            
def phaserate(hists,mids,exptime,bgb=False,bg=[],ens=['0.3-10 keV','S: 0.3-1.5 keV','H: 1.5-10 keV'],
              colors=['slateblue','maroon','navy','cornflowerblue'],figsize=(8,6),rate=False,
              ls=['solid','solid','solid','solid','solid','solid','solid'],enhist=False,ed=[],pd=pd,enbg=[],norm_bg=[]):
    '''
    Plots step histogram of phase, with count rate on y-axis
    
    Inputs:
    -------
    hists: list of counts hists (from phasehist_sh)
    mids
    exptime
    
    Optional Inputs:
    ----------------
    bgb: boolean for whether or not to subtract background (default False)
    bg: list of background counts to subtract from each bin for each hist
        values in count rate
    ens
    colors
    figsize
    rate: boolean for whether to include hardness ratio plot (default False)
          assumes indices of hard and soft to be same as in default ens
    ls: linestyles
    
    Outputs: 
    --------
    ratehists
    errs
    
    TO DO:
    ------
    fix linestyles
    '''
    #calculate count rate
    if enhist:
        fig,ax = plt.subplots(3,1,figsize=figsize,gridspec_kw = {'height_ratios':[3,4,3]})
        plt.subplots_adjust(wspace=0, hspace=0.1)
    elif rate:
        fig,ax = plt.subplots(2,1,figsize=figsize,sharex=True)
        plt.subplots_adjust(wspace=0, hspace=0.05)
    else:
        fig,ax = plt.subplots(figsize=figsize)
        
    ratehists = []
    errs=[]
    bins = len(hists[0][0]) #bins variable is total length, which is the double the number of phase bins
    for h in range(len(hists)):
        #calculate count rate from counts in each bin, #bins, exptime
        counts = hists[h][0] #array of counts in each bin
        binb = hists[h][1] #array of bin boundaries
        if bgb: cr = (counts*bins/(2*exptime))-bg[h]
        else: cr = (counts*bins/(2*exptime))
        if rate: hist = ax[0].hist(binb[:-1],binb,weights=cr,color=colors[h],histtype='step',linestyle=ls[h],label=ens[h])
        else: hist = ax.hist(binb[:-1],binb,weights=cr,color=colors[h],histtype='step',linestyle=ls[h],label=ens[h])
        #error is square root of counts divided by time in bin
        err = np.sqrt(counts)*bins/(2*exptime) #factor of 2 since really only 1/2 the data bins; just repeated
        if rate: 
            ax[0].errorbar(mids,hist[0],yerr=err,linestyle='none',color=colors[h])
            ax[0].set_ylabel('Counts/s',fontsize=18)
            ax[0].set_xlim(0,2)
            #specific to this dataset:
            if enhist:
                ax[0].set_ylim(0,14)
                ax[0].set_yticks(np.arange(0,16,2))
                ax[0].xaxis.set_major_locator(MultipleLocator(0.5))
                ax[0].xaxis.set_minor_locator(MultipleLocator(0.1))
        else: 
            ax.errorbar(mids,hist[0],yerr=err,linestyle='none',color=colors[h])
            ax.set_ylabel('Counts/s',fontsize=14)
        ratehists.append(hist)
        errs.append(err)
    plt.xlim(0,2)
    plt.xlabel('Phase',fontsize=18)
    if rate: ax[0].legend(loc='lower right') #fixing for now
    else: ax.legend()
    if enhist: rloc = 2
    else: rloc = 1
    if rate: #adds plot of hardness ratio below pulse profile
        softr = ratehists[1][0] #for now assumes order of soft and hard; should be generalized
        hardr = ratehists[2][0]
        se = np.sqrt(hists[1][0])*bins/(2*exptime)
        he = np.sqrt(hists[2][0])*bins/(2*exptime)
        serr=unumpy.uarray(softr,se)
        herr=unumpy.uarray(hardr,he)
        hre=(herr-serr)/(herr+serr)
        rerr=np.zeros(len(hre))
        hr=np.zeros(len(hre))
        for r in range(len(hre)):
            rerr[r]=hre[r].s 
            hr[r]=hre[r].n
        hrbins = hists[0][1]
        hrhist = ax[rloc].hist(hrbins[:-1],hrbins,weights=hr,color='slateblue',histtype='step',label='(H-S)/(H+S)')
        ax[rloc].errorbar(mids,hr,yerr=rerr,linestyle='none',color='slateblue')
        ax[rloc].legend()
        ax[rloc].set_ylabel('Hardness Ratio',fontsize=18)
        if enhist:
            ax[rloc].set_ylim(-.2,.15)
            ax[rloc].set_yticks(np.arange(-.2,.2,.05))
            ax[rloc].xaxis.set_major_locator(MultipleLocator(0.5))
            ax[rloc].xaxis.set_minor_locator(MultipleLocator(0.1))
        #add hardness ratio errors  
    #puts bg-subtracted, normalized, smoothed energy-phase heat map
    if enhist:
#         #need these defined
#         ph = ed['TIME']%pd
#         ph1 = ph/pd
#         ph2 = ph1 + 1
#         ph3 = list(ph1)+list(ph2)
#         ed2 = list(ed['PI'])+list(ed['PI'])
#         enhist = ax[1].hist2d(ph3,np.array(ed2)/100,bins=32) #convert from PI to keV
#         #normalizing and background-subtracting
#         norm_bg = np.zeros((32,32)) #16 phase bins
#         #enbg defined in jupyter notebook
#         for i in range(32):
#             norm_bg[i,:] = ((enhist[0][:,31-i]*16/exptime)-enbg[31-i,1]) #-enbg[i,1]
#             norm_bg[i,:] = norm_bg[i,:]/np.median(norm_bg[i,:]) #16 phase bins
        im = ax[1].imshow(norm_bg,cmap='plasma',interpolation='gaussian',aspect='auto')
        ax[1].set_xticks([])
        #ax[1].set_xticks([-0.5,-.5+31/8,-.5+62/8,-.5+93/8,-.5+124/8,-.5+155/8,-.5+186/8,31.5-31/8,31.5])
        #just on multiples of 0.5 and then minor ticks
        ax[1].set_xticks([-0.5,-.5+32/4,-.5+64/4,-.5+96/4,31.5])
        a=ax[1].get_xticks().tolist()
        #a=['0','0.25','0.5','0.75','1.0','1.25','1.50','1.75','2.0']
        a=['0','0.5','1.0','1.50','2.0']
        ax[1].set_xticklabels(a)
        ax[1].set_xticks(np.arange(-.5,31.5,32/20), minor=True)
        #trying to add minor ticks
        #ax[1].xaxis.set_major_locator(MultipleLocator(8))
#         ax[1].xaxis.set_major_formatter(FormatStrFormatter('%d'))
#         #ax[1].xaxis.set_minor_locator(MultipleLocator(1.6))
#         a=ax[1].get_xticks().tolist()
#         a=['0','0','0.5','1.0','1.5','2.0']
#         ax[1].set_xticklabels(a)
        
        ax[1].set_yticks([28.5,28.5-32/7.7,28.5-2*32/7.7,28.5-3*32/7.7,28.5-4*32/7.7,28.5-5*32/7.7,28.5-6*32/7.7,28.5-7*32/7.7])
        a=ax[1].get_yticks().tolist()
        a=['1','2','3','4','5','6','7','8']
        ax[1].set_yticklabels(a)
        ax[1].set_ylabel('Energy (0.3-8.0 keV)',fontsize=18)
        #fig.colorbar(im,ax=ax[1],label='Normalized Counts')
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(ax[1],
                   width="5%",  # width = 5% of parent_bbox width
                   height="100%",  # height : 50%
                   loc='lower left',
                   bbox_to_anchor=(1.02, 0., 1, 1),
                   bbox_transform=ax[1].transAxes,
                   borderpad=0
                   )
        
        cbar = fig.colorbar(im,cax=axins,orientation='vertical',pad=.01)
        #plt.colorbar().set_label(label='a label',size=15,weight='bold')
        cbar.set_label(label='Normalized Counts',size=18)
        #cbar.ax.tick_params() 
    return ratehists,errs


def s_phaserate(hist,mids,exptime,color='slateblue',bins=16): #16 bins means 16 phase bins (32 total)
    '''
    Plots single phase-folded step hist, with count rate on y-axis
    
    Inputs:
    -------
    hist: first output of phasehist
    mids: second output of phasehist
    exptime
    
    Optional Inputs:
    ----------------
    color
    bins: # of phase bins (note that this value different from other bin args
    
    Outputs:
    --------
    hist
    err
    '''
    counts = hist[0] #array of counts in each bin
    binb = hist[1] #array of bin boundaries
    cr = counts*bins/exptime
    hist = plt.hist(binb[:-1],binb,weights=cr,color=color,histtype='step')
    err = np.sqrt(counts)*bins/exptime
    plt.errorbar(mids,cr,yerr=err,color=color,linestyle='none')
    plt.xlim(0,2)
    plt.xlabel('Phase',fontsize=14)
    plt.ylabel('Counts/s',fontsize=14)
    return hist,err


def phaseroll(bins,weights,by,mids,err=0,figsize=(8,6),color='slateblue',label=''):
    '''
    Roll/shift histogram by some number of bins
    
    Inputs:
    -------
    bins
    weights
    by: how many bins to shift by (end goes back to beginning
    Optional Inputs:
    ----------------
    err: array of errors for bins
    figsize
    color
    '''
    ws = np.roll(weights,by) #shifted weights by some number of bins
    e = np.roll(err,by) #haven't tested shifted error
    plt.figure(figsize=figsize)
    hist = plt.hist(bins[:-1],bins,weights=ws,color=color,histtype='step',label=label)
    plt.xlim(0,2)
    plt.xlabel('Phase',fontsize=14)
    plt.ylabel('Counts/s',fontsize=14)
    plt.errorbar(mids,ws,yerr=e,color=color,linestyle='none')
    return hist
    
    
def binpi(evt_data,secs=10):
    '''
    Bins event data by given time interval by starting a new bin
         if the time of an event is greater than the previous start
         bin time minus secs.
         
    Inputs:
    -------
    evt_data: astropy table of events
    secs: int or float of time-width of bins in seconds
          default 10 seconds
    
    Outputs:
    --------
    bins: array of indices in evt_data of first event in bin
    times: array of mean time of each bin
    counts: array of total counts in each bin
    '''
    bins = [0]
    times = []
    c=0
    for e in range(len(evt_data)):
        if evt_data['TIME'][e]-evt_data['TIME'][bins[c]]>secs:
            c+=1
            bins.append(e)
            times.append(np.mean((evt_data['TIME'][e-1],evt_data['TIME'][bins[-2]])))
    counts = []
    for b in range(1,len(bins)):
        counts.append(bins[b]-bins[b-1])
    return np.array(bins), np.array(times), np.array(counts)

def phasescatter(times,counts,pd,figsize=(10,8),color='navy'):
    '''
    Plots phase-folded lightcurve using pre-binned data.
    
    Inputs:
    -------
    times: array of mean times of bins
    counts: counts per bin
    pd: period used for phase-folding
    figsize: tuple of matplotlib figuresize
             default (8,6)
    color: matplotlib color of hist
           default navy
    '''
    
    plt.figure(figsize=figsize)
    plt.errorbar(times%pd,counts,yerr=np.sqrt(counts),linestyle='none',marker='o',color=color)
    plt.xlabel('Time (s)',fontsize=14)
    plt.ylabel('Counts',fontsize=14)            

    
def ls(t,c,lowf=.01,highf=1,spp=10,ylim=(0,.5),xlow=0):
    '''
    Plots Lomb-Scargle Periodogram and return top 10
    highest-powered periods
    
    Inputs:
    -------
    t:
    c:
    lowf: minimum_frequency to search
    highf: maximum frequency to search
    spp: samples per peak
    ylim: tuple of limits of plot
    xlow: int or float of lower end of x limit of plot
    
    Outputs:
    --------
    first 10 values of DataFrame containing frequencies, periods,
    and power in descending order by power
    '''
    fig = plt.figure(figsize=(10, 3))
    gs = plt.GridSpec(2, 2)

    dy = np.sqrt(c)

    ax = fig.add_subplot(gs[:, 0])
    ax.errorbar(t, c, dy, fmt='ok', ecolor='gray',
            markersize=3, capsize=0)
    ax.set(xlabel='time',
       ylabel='signal',
       title='Data and Model')

    ls = LombScargle(t, c)
    freq, power = ls.autopower(normalization='standard',
                           minimum_frequency=lowf,
                           maximum_frequency=highf,
                           samples_per_peak=spp)

    plt.plot(1./freq,power,color='rebeccapurple')
    plt.xlabel('Period (s)')
    plt.ylabel('Power')
    plt.xlim(xlow,1/lowf)
    plt.ylim(ylim)


    best_freq = freq[np.argmax(power)]
    print(1./best_freq)
    frame = pd.DataFrame(columns=['f','pow','pd'])
    frame['f'] = freq
    frame['pow'] = power
    frame['pd'] = 1./freq
    return(frame.sort_values(by='pow',ascending=False)[:10])

def detrend(c, window=81): #Georgios' code from OGLE LXP 69.5 notebook
    '''
    Detrends counts values.
    
    Inputs:
    -------
    c: counts per bin
    window: must be odd
    
    Outputs:
    --------
    the detrended counts plus the mean count value
    '''
    c = np.array(c)
    print('Smooth (window = ', window, ') and detrend data...')
    csmooth = signal.savgol_filter(c, window, 1)
    cmean = np.mean(c)
    return c - csmooth  + c.mean()

def stingphase(evt_data,freq,figsize=(10,10),nbins=32,save=False,pname='stingphase',mjd=False):
    '''
    Plots Stingray phaseogram.
    Inputs:
    -------
    evt_data
    freq
    
    figsize
    nbins: number of bins for histogram and phaseogram
    save: bool for whether or not to save plot as png
    pname: string of path and name of output png file to save
    
    TO DO:
    add option to put MJD referencce time
    '''
    plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=(1, 3))
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex=ax0)
    plt.subplots_adjust(wspace=0, hspace=0)
    phaseogr, phases, times, additional_info = \
            phaseogram(evt_data['TIME'], freq, return_plot=True,nt=nbins,nph=nbins)
    mean_phases = (phases[:-1] + phases[1:]) / 2
    plot_profile(mean_phases, np.sum(phaseogr, axis=1), ax=ax0)
    _ = plot_phaseogram(phaseogr, phases, times, ax=ax1, vmin=np.median(phaseogr))
    if save:
        plt.savefig(pname+'.png',dpi=200,bbox_inches='tight')
    
    
    