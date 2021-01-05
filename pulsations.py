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
from scipy.ndimage.filters import gaussian_filter
import scipy.stats as st


def readevt(file,ind=1):
    '''
    Reads evt file and returns astropy Table
    '''
    hdu_list = fits.open(file, memmap=True)
    return Table(hdu_list[ind].data)

def phasehist(evt_data,pd,figsize=(8,6),color='slateblue',bins=32,epoch=0):
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
    bins: int of total bins to use in hist
          default 32 (16 phase bins)
          
    Outputs:
    --------
    a: histogram: array where 0 index array is counts in each bin;
                  index 1 array is bin boundaries
    mids: list of phase bin centers
    '''
    plt.figure(figsize=figsize)
    ph = (evt_data['TIME']+epoch)%pd 
    ph_1 = ph/pd #divides phase by period so that x-axis goes to 1, not pd
    ph_2 = ph_1+1 
    ph_3 = list(ph_1)+list(ph_2) #duplicates and concatenates 
    binw = 2./bins
    a = plt.hist(ph_3,bins=np.arange(0,2.01,binw),color='slateblue',histtype='step',label='all')
    plt.xlim(0,2)
    plt.xlabel('Phase',fontsize=14)
    plt.ylabel('Counts',fontsize=14)
    #constructs list of middle of each phase bin
    mids = []
    for t in range(1,len(a[1])):
        mids.append(np.median((a[1][t],a[1][t-1])))
    #adds error as sqrt of counts
    plt.errorbar(mids,a[0],yerr=np.sqrt(a[0]),linestyle='none',color=color)
    #returns histogram (counts and bin boundary arrays), and array of middles of phase bins
    return a,mids

def phasehist_sh(ed,pd,ens=['0.3-10 keV','0.3-1.5 keV','1.5-10 keV'],colors=['slateblue','palevioletred','navy','cornflowerblue'],figsize=(8,6),bins=32,epoch=0):
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
        ph = (ed[e]['TIME']+ epoch)%pd
        ph_1 = ph/pd
        ph_2 = ph_1+1
        ph_3 = list(ph_1)+list(ph_2)
        #define bins with bin width rather than total bin number
        binw = 2./bins
        hist = plt.hist(ph_3,bins=np.arange(0,2.01,binw),color=colors[e],histtype='step',label=ens[e])
        hists.append(hist)
        
        if e == 0:
            mids = []
            for t in range(1,len(hist[1])):
                mids.append(np.median((hist[1][t],hist[1][t-1])))
        plt.errorbar(mids,hist[0],yerr=np.sqrt(hist[0]),linestyle='none',color=colors[e])

    plt.xlim(0,2)
    plt.xlabel('Phase',fontsize=14)
    plt.ylabel('Counts',fontsize=14)
    plt.legend()
    return hists,mids
            
def phaserate(hists,mids,exptime,bgb=False,bg=[],ens=['0.3-10 keV','S: 0.3-1.5 keV','H: 1.5-10 keV'],
              colors=['slateblue','maroon','navy','cornflowerblue'],figsize=(8,6),rate=False,
              ls=['solid','solid','solid','solid','solid','solid','solid'],enhist=False,ed=[],pd=pd,enbg=[],norm_bg=[],
              yrange1=[0,14],yrange2=[-1,1],rethr=False,title=''):
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
    
    rethr: boolean determining if hardness ratio values returned
    title: string for fig title; used if length > 0 
    
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
        fig,ax = plt.subplots(3,1,figsize=figsize,gridspec_kw = {'height_ratios':[4,3,3]})
        plt.subplots_adjust(wspace=0, hspace=0.05)
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
        if enhist: ploc = 1
        else: ploc = 0
        if rate: hist = ax[ploc].hist(binb[:-1],binb,weights=cr,color=colors[h],histtype='step',linestyle=ls[h],label=ens[h])
        else: hist = ax.hist(binb[:-1],binb,weights=cr,color=colors[h],histtype='step',linestyle=ls[h],label=ens[h])
        #error is square root of counts divided by time in bin
        err = np.sqrt(counts)*bins/(2*exptime) #factor of 2 since really only 1/2 the data bins; just repeated
        if rate: 
            ax[ploc].errorbar(mids,hist[0],yerr=err,linestyle='none',color=colors[h])
            ax[ploc].set_ylabel('Rate (c/s)',fontsize=18)
            ax[ploc].set_xlim(0,2)
            ax[ploc].set_xticks(np.arange(0,2,.5))
            if enhist: ax[ploc].set_xticklabels(['','','','',''])
            ax[ploc].set_ylim(yrange1)
            ax[ploc].set_yticks(np.arange(yrange1[0],yrange1[1]+1.5,2))
            ax[ploc].minorticks_on()

        else: 
            ax.errorbar(mids,hist[0],yerr=err,linestyle='none',color=colors[h])
            ax.set_ylabel('Rate (c/s)',fontsize=16)
        ratehists.append(hist)
        errs.append(err)
    plt.xlim(0,2)
    plt.xlabel('Phase',fontsize=16)
    if rate: ax[ploc].legend(loc='upper right') #fixing for now
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
        hrhist = ax[rloc].hist(hrbins[:-1],hrbins,weights=hr,color='slateblue',histtype='step')
        ax[rloc].errorbar(mids,hr,yerr=rerr,linestyle='none',color='slateblue')
        #ax[rloc].legend()
        ax[rloc].set_ylabel('HR',fontsize=18)
        ax[rloc].set_ylim(yrange2)
        ax[rloc].set_yticks(np.arange(yrange2[0],yrange2[1]+0.03,.1))
        ax[rloc].set_xticks(np.arange(0,2.1,0.5))
        ax[rloc].minorticks_on()
        #add hardness ratio errors
        
        #use horizontal shading for mean HR and error
        mean_hr = np.mean(hr)
        mean_hr_err = st.sem(hr)
        ax[rloc].axhspan(mean_hr-mean_hr_err,mean_hr+mean_hr_err,color='slateblue',alpha=.4)
    #puts bg-subtracted, normalized, smoothed energy-phase heat map
    if enhist:
        norm_bg_fil=gaussian_filter(norm_bg, 0.8,mode='wrap')
        im = ax[0].imshow(norm_bg_fil,cmap='plasma',interpolation='spline16',aspect='auto')
        #ax[1].set_xticks([-0.5,-.5+31/8,-.5+62/8,-.5+93/8,-.5+124/8,-.5+155/8,-.5+186/8,31.5-31/8,31.5])
        #just on multiples of 0.5 and then minor ticks
        ax[0].set_xticks([-0.5,-.5+32/4,-.5+64/4,-.5+96/4,31.5])
        a=ax[0].get_xticks().tolist()
        #a=['0','0.25','0.5','0.75','1.0','1.25','1.50','1.75','2.0']
        a=['','','','','']
        ax[0].set_xticklabels(a)
        ax[0].set_xticks(np.arange(-.5,31.5,32/20), minor=True)
        ax[0].set_yticks([28.5,28.5-32/7.7,28.5-2*32/7.7,28.5-3*32/7.7,28.5-4*32/7.7,28.5-5*32/7.7,28.5-6*32/7.7,28.5-7*32/7.7])
        a=ax[0].get_yticks().tolist()
        a=['1','2','3','4','5','6','7','8']
        ax[0].set_yticklabels(a)
        ax[0].set_ylabel('Energy (0.3-8.0 keV)',fontsize=18)
        #fig.colorbar(im,ax=ax[1],label='Normalized Counts')
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(ax[0],
                   width="100%",  # width = 5% of parent_bbox width
                   height="5%",  # height : 50%
                   loc='upper right',
                   bbox_to_anchor=(0.0, 0.07, 1, 1),
                   bbox_transform=ax[0].transAxes,
                   borderpad=0
                   )
        
        cbar = fig.colorbar(im,cax=axins,orientation='horizontal',pad=.02)
        axins.xaxis.set_label_position('top')
        axins.xaxis.set_ticks_position('top')
        #plt.colorbar().set_label(label='a label',size=15,weight='bold')
        cbar.set_label(label='Normalized Counts',size=18)
        #cbar.ax.tick_params() 
    if len(title)>0: ax[0].set_title(title,fontsize=18)
        
    if rethr: return [hr,rerr],ratehists,errs
    else: return ratehists,errs


def s_phaserate(hist,mids,exptime,color='slateblue',bins=32,label='',scale=1,bgb=False,bg=[]): #16 bins means 16 phase bins (32 total)
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
    bgb: whether background included for background subtraction
    bg:
    
    Outputs:
    --------
    hist
    err
    '''
    counts = hist[0]*scale #array of counts in each bin multiplied by optional scale factor
    binb = hist[1] #array of bin boundaries
    if bgb and scale ==1: cr = (counts*bins/(exptime))-bg
    elif scale == 1: cr = counts*bins/exptime
    else: cr = counts
    hist2 = plt.hist(binb[:-1],binb,weights=cr,color=color,histtype='step',label=label)
    if scale == 1: err = np.sqrt(counts)*bins/exptime
    else: err = scale*np.sqrt(hist[0])
    plt.errorbar(mids,cr,yerr=err,color=color,linestyle='none')
    plt.xlim(0,2)
    plt.xlabel('Phase',fontsize=14)
    plt.ylabel('Counts/s',fontsize=14)
    return hist2,err


def phaseroll(bins,weights,by,mids,err=0,figsize=(8,6),color='slateblue',label='',ls='solid',new=True,reterr=False):
    '''
    Roll/shift histogram by some number of bins
    
    Inputs:
    -------
    bins
    weights
    by: how many bins to shift by (end goes back to beginning)
    Optional Inputs:
    ----------------
    err: array of errors for bins
    figsize
    color
    '''
    ws = np.roll(weights,by) #shifted weights by some number of bins
    e = np.roll(err,by) #haven't tested shifted error
    if new: plt.figure(figsize=figsize)
    hist = plt.hist(bins[:-1],bins,weights=ws,color=color,histtype='step',label=label,ls=ls)
    plt.xlim(0,2)
    plt.xlabel('Phase',fontsize=16)
    plt.ylabel('Counts/s',fontsize=14)
    plt.errorbar(mids,ws,yerr=e,color=color,linestyle='none')
    if reterr: return hist,e
    else: return hist
    
    
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
            times.append(np.median((evt_data['TIME'][e-1],evt_data['TIME'][bins[-2]])))
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
    
    
def pf(evt,pd,expt,bg=True,bgs=[],scale=1,guess=False,gf=0.8,retearly=False,onehard=False,nbins=64,plot=False):
    '''Make separate event tables for each energy band used in PF vs. energy fig
    onehard: only one bin from 3.0-8.0 keV
    nbins: total number of bins; so #phase bins is half of value given'''
    #make separate tables for each energy band
    en1 = evt[evt['PI']<=70]
    en2 = evt[evt['PI']>70]
    en2 = en2[en2['PI']<=100]
    en3 = evt[evt['PI']>100]
    en3 = en3[en3['PI']<=150]
    en4 = evt[evt['PI']>150]
    en4 = en4[en4['PI']<=200]
    en5 = evt[evt['PI']>200]
    en5 = en5[en5['PI']<=300]
    en6 = evt[evt['PI']>300]
    #only one bin 3-8 keV
    if onehard: en6 = en6[en6['PI']<=800]
    #two bins 3-8 keV
    else:
        en6 = en6[en6['PI']<=500]
        en7 = evt[evt['PI']>500]
        en7 = en7[en7['PI']<=800]
    #how much to subtract from each bin
    
    if bg:
        bg1 = bgs[0]*len(en1)/expt #avg rate * percentage that's BG
        bg2 = bgs[1]*len(en2)/expt
        bg3 = bgs[2]*len(en3)/expt
        bg4 = bgs[3]*len(en4)/expt
        bg5 = bgs[4]*len(en5)/expt
        bg6 = bgs[5]*len(en6)/expt
        if not onehard: 
            bg7 = bgs[6]*len(en7)/expt
            pfbg = scale*np.array([bg1,bg2,bg3,bg4,bg5,bg6,bg7])
        else: 
            pfbg = scale*np.array([bg1,bg2,bg3,bg4,bg5,bg6])
    #energy list of labels whether or not bg-subtraction
    if onehard:
        ens=['0.3-0.7','0.7-1.0','1.0-1.5','1.5-2.0','2.0-3.0','3.0-8.0']
        #list of evt tables
        entab = [en1,en2,en3,en4,en5,en6]
    else: 
        ens=['0.3-0.7','0.7-1.0','1.0-1.5','1.5-2.0','2.0-3.0','3.0-5.0','5.0-8.0']
        entab = [en1,en2,en3,en4,en5,en6,en7]

    #counts histogram
    pfens,pfmids = phasehist_sh(entab,pd,ens=ens,bins=nbins,
               colors=['palevioletred','rebeccapurple','cornflowerblue','royalblue','navy','darkseagreen','darkgreen'],figsize=(5,4))
    if not plot: plt.close()
    if guess:
        #find CRs without bg-subtraction
        pfcr,pfcrerr = phaserate(pfens,pfmids,expt,ens=ens,
               colors=['palevioletred','rebeccapurple','cornflowerblue','royalblue','navy','darkseagreen','darkgreen'],
                           rate=False,figsize=(4,2))
        if not plot: plt.close()
        bgguess = gf*np.min(pfcr[-1][0])/bg7
        #guess background for epoch 1
        print(bgguess)
        pfbg = bgguess*np.array(pfbg)
    
    #count rate histograms
    if bg: 
        pfcr,pfcrerr = phaserate(pfens,pfmids,expt,ens=ens,
               colors=['palevioletred','rebeccapurple','cornflowerblue','royalblue','navy','darkseagreen','darkgreen'],
                           rate=False,figsize=(7,4),bgb=True,bg=pfbg)
    else: 
        #no bg-subtraction
        pfcr,pfcrerr = phaserate(pfens,pfmids,expt,ens=ens,
               colors=['palevioletred','rebeccapurple','cornflowerblue','royalblue','navy','darkseagreen','darkgreen'],
                           rate=False,figsize=(7,4))
    if not plot: plt.close()
    if retearly: return pfcr,pfcrerr
    #calculated pulsed fractions
    #with bg-subtraction
    pfr = [] #PF and PF error
    pfnr = [] #just PF
    pfer = [] #just PF error
    if onehard: ennum = 6
    else: ennum = 7
    for i in range(ennum):
        cr = pfcr[i][0]
        totlen = len(cr)
        r = cr[:int(totlen/2)] #first half of count rates, since second half just repeated
        e = pfcrerr[i][:int(totlen/2)] #errors originally from sqrt of counts
        rerr = unumpy.uarray(r,e)
        rmax = rerr.max()
        rmin = rerr.min()
        pfr.append((rmax-rmin)/(rmax+rmin))
        pfnr.append(((rmax-rmin)/(rmax+rmin)).n) #value
        pfer.append(((rmax-rmin)/(rmax+rmin)).s) #propogated error
    #return PFs and PF errors separately
    return pfnr,pfer

def spf(evt,pd,expt,bg=True,sbg=0,scale=1,guess=False,gf=0.8,retearly=False,nbins=64):
    '''Calculate PF for 0.3-8.0 band'''
    pfbg = scale*sbg*len(evt)/expt
    pfens,pfmids = phasehist(evt,pd,figsize=(4,3),bins=nbins) #32 phase bins
    plt.close()
    if bg: 
        #bins argument in s_phaserate is number of phase bins
        pfcr,pfcrerr = s_phaserate(pfens,pfmids,expt,bins=nbins/2,bgb=bg,bg=pfbg)
        plt.close()
    #return if using rms PF instead
    if retearly: return pfcr[0],pfcrerr
    #calculation of single PF value
    cr = pfcr[0]
    totlen = len(cr)
    r = cr[:int(totlen/2)] #first half of count rates, since second half just repeated
    e = pfcrerr[:int(totlen/2)] #errors originally from sqrt of counts
    rerr = unumpy.uarray(r,e)
    rmax = rerr.max()
    rmin = rerr.min()
    pfr = (rmax-rmin)/(rmax+rmin)
    pfnr = ((rmax-rmin)/(rmax+rmin)).n #value
    pfer = ((rmax-rmin)/(rmax+rmin)).s #propogated error
    #return PFs and PF errors separately
    return pfnr,pfer

def rms_pf(evt,pd,expt,bg=True,bgs=[],scale=1,guess=False,onehard=False,nbins=64,onephase=False,printall=False):
    '''Calculate PFs using rms definition'''
    pfcr,pfcrerr = pf(evt,pd,expt,bg=bg,bgs=bgs,scale=scale,guess=guess,retearly=True,onehard=onehard,nbins=nbins)
    rms_pf = []
    rms_pfe = []
    if onehard: ennum = 6
    else: ennum = 7
    for i in range(ennum):
        cr = pfcr[i][0]
        e = pfcrerr[i]
        #call function below to get rmf PF for each band
        rms_pf.append(rms_spf(cr=cr,err=e,onephase=onephase,printall=printall).n)
        rms_pfe.append(rms_spf(cr=cr,err=e,onephase=onephase).s)
    return rms_pf,rms_pfe
        
def rms_spf(cr=0,err=0,evt=0,pd=0,expt=0,sbg=0,getsingle=False,nbins=64,onephase=True,printall=False):
    '''Calculate single rms PF'''
    if getsingle: cr,err = spf(evt,pd,expt,bg=True,sbg=sbg,retearly=True,nbins=nbins)
    totlen = len(cr)
    if printall: print('tot len:',totlen)
    if onephase: #only use up to phase 1 in calculation
        cr=cr[:int(totlen/2)] #count rate in phase bins up to 1
        err=err[:int(totlen/2)]
    rerr=unumpy.uarray(cr,err)
    #phase-averaged count rate
    avgr = np.mean(rerr)
    #len(r) is N (number of phase bins)
    num = np.sum(((rerr-avgr)**2))/len(cr)
    if printall: print(len(cr))
    num = (num)**(1/2)
    full = num/avgr
    return full
