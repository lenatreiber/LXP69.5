from uncertainties import ufloat
from uncertainties.umath import *
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.stats import LombScargle
from scipy import signal
import warnings
warnings.filterwarnings("ignore")
import scipy.optimize
from lmfit.models import GaussianModel
from numpy import exp, pi, sqrt
from lmfit import Model
import scipy.optimize
import sklearn.utils as sk
from scipy.optimize import curve_fit


#----------------------GAUSSIAN FIT AND GENERAL PLOTTING----------------------
def gparams(data,inds,offset=0,mags=1,more=False):
    '''
    Fits Gaussian to each of 13 flares. 
    Inputs:
    -------
    data: DataFrame with 'MJD-50000' and 'I mag' columns (sog4 in notebook)
    inds: list of lists of index bounds of flares to fit (flareinds in notebook)
    offset: add to the difference between the flare's mag maximum and the data (default 0) 
    mags: ind of I data to use; 1 is original, 5 is detrended
    more: if True, returns both list of fitting results and DataFrame of best-fit parameters for each flare
        default False, returning just the DataFrame
    
    Outputs:
    --------
    if more:
        gouts: list of model results
        ggfits: DataFrame with one row per flare, including the best fit parameter values
    else:
        ggfits
    '''
    gouts = []
    for f in range(len(inds)):   
        st,end = inds[f][0],inds[f][1]
        #x = data[st:end,time]
        x = np.array(data['MJD-50000'][st:end])
        #uses original (not detrended) data
        #y = offset + np.max(data[st:end,mags]) - data[st:end,mags]
        if mags == 1: imag = np.array(data['I mag'][st:end])
        else: imag = np.array(data['I detrend 2'][st:end])
        y = np.max(imag) - imag + offset
        mod = GaussianModel()
        pars = mod.guess(y, x=x)
        out = mod.fit(y, pars, x=x)
        gouts.append(out)
    #DataFrame with values for each flare; includes error columns but those have to be hard-coded?
    ggfits = pd.DataFrame(columns=['center','sigma','fwhm','height','amp'])
    #13 flares
    ggfits['center'] = np.zeros(13)
    i = 0
    for o in gouts: 
        ggfits['center'][i] = o.params['center'].value
        ggfits['fwhm'][i] = o.params['fwhm'].value
        ggfits['height'][i] = o.params['height'].value
        ggfits['amp'][i] = o.params['amplitude'].value
        ggfits['sigma'][i] = o.params['sigma'].value
        i += 1
    if more: return gouts,ggfits
    else: return ggfits
    
def plotrel(gfl,colors=['navy','cornflowerblue','rebeccapurple','palevioletred','darkseagreen']): #gaussian fit list
    '''Plots relations in scatter subplots between Gaussian model quantities.
    Inputs:
    -------
    gfl: list of DataFrames with center, height, fwhm, amp
    colors: list of colors to use for each DataFrame in gfl; must be at least the same length
    '''
    fig3 = plt.figure(constrained_layout=True,figsize=(11,9))
    gs = fig3.add_gridspec(3, 3)
    ax1 = fig3.add_subplot(gs[0, 0])
    ax2 = fig3.add_subplot(gs[0, 1],sharey=ax1)
    ax3 = fig3.add_subplot(gs[0, 2],sharey=ax1)
    ax4 = fig3.add_subplot(gs[1, 0])
    ax5 = fig3.add_subplot(gs[1, 1],sharey=ax4)
    ax6 = fig3.add_subplot(gs[2, 1])
    #colors = ['navy','cornflowerblue','rebeccapurple','palevioletred','darkseagreen']
    i = 0
    for g in gfl:
        ax1.errorbar(g['amp'],g['height'],linestyle='none',marker='o',alpha=.7,color=colors[i])
        ax2.errorbar(g['center'],g['height'],linestyle='none',marker='o',alpha=.7,color=colors[i])
        ax4.errorbar(g['amp'],g['fwhm'],linestyle='none',marker='o',alpha=.7,color=colors[i])
        ax5.errorbar(g['center'],g['fwhm'],linestyle='none',marker='o',alpha=.7,color=colors[i])

        ax6.errorbar(g['center'],g['amp'],linestyle='none',marker='o',alpha=.7,color=colors[i])
        ax6.set_xlabel('Center',fontsize=14)
    
        ax3.errorbar(g['fwhm'],g['height'],linestyle='none',marker='o',alpha=.7,color=colors[i])
        i+=1
    #set labels once
    ax4.set_ylabel('FWHM',fontsize=14)
    ax1.set_ylabel('Height',fontsize=14)
    ax1.set_xlabel('Amplitude',fontsize=14)
    ax6.set_ylabel('Amplitude',fontsize=14)
    ax2.set_xlabel('Center',fontsize=14)
    ax4.set_xlabel('Amplitude',fontsize=14)
    ax5.set_xlabel('Center',fontsize=14)
    ax3.set_xlabel('FWHM',fontsize=14)
    
def plflares(result,cens,inds,data,offset=0,modcolor='navy',cs = ['navy','cornflowerblue','rebeccapurple','palevioletred','darkseagreen','forestgreen','maroon','salmon','gold'],label=False,labels=[''],ylabel='I max mag - I mag + offset'):
    '''
    Plots 13 flares with one model and multiple centers shown.
    
    Inputs: 
    -------
    result: list of outputs from gaussian model to be shown
    cens: centers to plot as vertical lines
    inds: index bounds of flares (flareinds)
    data: DataFrame for plotting (sog4)
    offset: vertical offset from max I mag - I mag to use in plotting
    modcolor: color of model curve
    cs: colors of cens
    label: whether to label different center lists (default False)
    labels: list of strings to use for labels; same order as cens
    ylabel: label for y-axis
    '''
    fig = plt.figure(constrained_layout=True,figsize=(18,10))
    gs = fig.add_gridspec(3,6)
    
    for f in range(13):
        if f < 6: axt = fig.add_subplot(gs[0, f])
        #second row of plots
        elif f < 12: axt = fig.add_subplot(gs[1, f-6])
        #final flare in third row
        else: axt = fig.add_subplot(gs[2, f-12])
        st = inds[f][0]
        end = inds[f][1]
        stdate = int(data['MJD-50000'][st:st+1])
        enddate = int(data['MJD-50000'][end-1:end])
        #match offset to offset used in model that is shown
        axt.scatter(data['MJD-50000'][st:end],np.max(data['I mag'][st:end])-data['I mag'][st:end]+offset,color='navy')
        xfits = np.linspace(enddate,stdate,enddate-stdate)
        bfit = gaussian(xfits,result[f].best_values['amplitude'],result[f].best_values['center'],result[f].best_values['sigma'])
        axt.plot(xfits,bfit,color=modcolor) 
        
        for i in range(len(cens)): #len(cens) is number of models of centers to plot
            if label: axt.axvline(cens[i][f],color=cs[i],label=labels[i])
            else: axt.axvline(cens[i][f],color=cs[i])
        
        if f == 0 and label: axt.legend()
    axt.set_ylabel(ylabel,fontsize=16)
    axt.set_xlabel('MJD-50000',fontsize=16)
    


def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid) from lmfits"""
    return (amp / (sqrt(2*pi) * wid)) * exp(-(x-cen)**2 / (2*wid**2))


def line(x, slope, intercept):
    """a line"""
    return slope*x + intercept

def gaussline(data,flareinds,ind,guess_out,plot=True,offset=0,mags=1,exguess=False,more=False):
    '''
    Fits gaussian+line composite model to the 13 flares.
    Inputs:
    -------
    data: sog4
    flareinds: flareinds
    ind: index of flare in flareinds to fit
    guess_out: gaussian output to use as a guess for the amp, cen, and wid
    plot: default True -- plots best fit on flare data
    offset: vertical offset 
    mags: original or detrended data (defaultt 1 for original data)
    exguess: default False; make True if guess_out is list of guess parameters rather than gaussian result
    more: returns result rather than best parameter values if True (default False)
    
    Output:
    -------
    if more: returns fit result
    else: returns best fit cen, amp, and wid
    '''
    x = data['MJD-50000'][flareinds[ind][0]:flareinds[ind][1]]
    #default mag 1 is original data; 5 is detrended
    if mags == 1: imag = data['I mag']
    else: imag = data['I detrend 2']
    y = offset + np.max(imag[flareinds[ind][0]:flareinds[ind][1]]) - imag[flareinds[ind][0]:flareinds[ind][1]]
    mod = Model(gaussian) + Model(line)
    #using guesses from gaussian model
    if exguess:
        amp,cen,wid=guess_out[0],guess_out[1],guess_out[2]
    else:
        gf = guess_out[ind].best_values #getting guesses from previous fit (just gaussian)
        amp, cen, wid = gf['amplitude'], gf['center'], gf['sigma']
    pars = mod.make_params(amp=amp, cen=cen, wid=wid, slope=0, intercept=1)
    pars['amp'].min = 0.0
    #center constrained to be within 30 days of .05 offset center
    pars['cen'].min = cen-30
    pars['cen'].max = cen+30
    #positive width
    pars['wid'].min = 0.0

    result = mod.fit(y, pars, x=x)
    if plot:
        plt.plot(x, y, 'bo')
        #plot best fit model and center
        plt.plot(x, result.best_fit, 'r-', label='best fit')
        plt.axvline(result.best_values['cen'],color='black')
        plt.legend(loc='best')
        plt.show()
    if more: return result
    else: return result.best_values

def flarehist(centers,refcenters,labels=['detrended','0.06 offset','gaussian+line','det gaussian+line','triangle'],bins=[-20,-15,-10,-5,0,5,10,15,20],
              colors=['navy','palevioletred','forestgreen','maroon','salmon','gold'],
              linestyles=['solid','dashed','dotted','-.',(0, (5, 10))]):
    '''
    Plots step histogram of the difference between input centers and a list of reference centers.
    Inputs:
    -------
    centers: list of list of centers for each flare
    refcenters: reference centers to subtract (equal length to a given list element of centers)
    labels: labels in same order as centers
    bins: bin bounds
    colors: colors to use in histogram
    linestyles: linestyles to use in hist
    '''
    all_diffs = []
    for c in range(len(centers)):
        diff = centers[c] - np.array(refcenters)
        all_diffs.append(diff)
        a = plt.hist(centers[c] - refcenters,label=labels[c],bins=bins,histtype='step',color=colors[c],linestyle=linestyles[c],alpha=.7)
    plt.legend(loc='upper left')
    plt.xlabel('Flare Center - Gaussian Flare Center')
    plt.ylabel('# Flares')
    #returns standard deviation
    return np.nanstd(all_diffs)

def oneflareg(data,st,end,gline=False,det=False,offset=0):
    '''Fits gaussian to same flare using different indices
    data: sog4
    st: list/array of start indices
    end: list/array of end indices
    gline: boolean for whether or not to use gauss+line model (default False)
    det: boolean for whether or not to use detrended data (default False)
    offset: vertical offset for fitting (default 0)
    Returns list of centers'''
    cens = np.zeros((len(st),len(end)))
    for i in range(len(st)):
        for j in range(len(end)):
            time = np.array(data['MJD-50000'][st[i]:end[j]])
            if det: imag = np.array(data['I detrend 2'][st[i]:end[j]])
            else: imag = np.array(data['I mag'][st[i]:end[j]])
            if gline:
                #make sure this guess is reasonable
                guesscen = np.float(data['MJD-50000'][st[i]]+(data['MJD-50000'][end[j]]-data['MJD-50000'][st[i]])/2)
                bvs = gaussline(data,[[st[i],end[j]]],0,[5,guesscen,30],offset=offset,plot=False,exguess=True)
                #bvs = gaussline(data,[[st[i],end[j]]],0,[5,5500,30],offset=offset,plot=False,exguess=True)
                cens[i][j] = bvs['cen']
            #regular gaussian, with no offset
            else:
                x = time
                y = np.max(imag) - imag + offset
                mod = GaussianModel()
                pars = mod.guess(y, x=x)
                out = mod.fit(y, pars, x=x)
                cens[i][j] = out.params['center'].value
                print('st:',st[i],'end:',end[j],'cen:',cens[i][j])
    return cens

    

def bootg(data,flareinds,ind,st=40,end=130,indiv=False,num=100,offset=0):
    '''Bootstrap gaussian fit for one flare.
    Inputs:
    -------
    data: sog4
    flareinds: flareinds
    ind: index of flare to bootstrap (used to get inds from flareinds if indiv False)
    st: used as start index if indiv True
    end: used as end index if indiv True
    indiv: use individual index arguments (st,end) rather than ind, which then references flareinds (default False)
    num: number of bootstrap iterations (default 100)
    offset: gaussian vertical offset
    
    Outputs:
    --------
    bsouts: list of bootstrap model results
    bsfits: DataFrame with parameters from each bootstrap iteration 
    '''
    if indiv:
        ind1,ind2 = st,end
    else:
        ind1,ind2 = flareinds[ind][0],flareinds[ind][1]
    bsouts = []
    for i in range(num):
        #bootstrap indices of first flare
        bs = sk.resample(np.arange(ind1,ind2))
        bst = np.array(data['MJD-50000'][bs])
        bsi = np.array(data['I mag'][bs])
        x = bst
        #uses original (not detrended) data
        y = np.max(bsi) - bsi + offset
        mod = GaussianModel()
        pars = mod.guess(y, x=x)
        out = mod.fit(y, pars, x=x)
        bsouts.append(out)
    bsfits = pd.DataFrame(columns=['center','sigma','fwhm','height','amp'])
    bsfits['center'] = np.zeros(num)
    #adding to DataFrame
    i = 0
    for b in bsouts: 
        bsfits['center'][i] = b.params['center'].value
        bsfits['fwhm'][i] = b.params['fwhm'].value
        bsfits['height'][i] = b.params['height'].value
        bsfits['amp'][i] = b.params['amplitude'].value
        bsfits['sigma'][i] = b.params['sigma'].value
        i += 1
    #returns list of model results and DataFrame with compiled best fit parameter values
    return bsouts,bsfits

#----------------------TRIANGULAR FIT----------------------
def triangle(x,s1,i1,s2,i2,findpk=False):
    '''
    Triangle function using two line slopes and two intercepts.
    Inputs:
    ------
    x: array or list of x values passed into triangle function
    s1: slope of first line
    i1: intercept of first line
    s2: slope of second line
    i2: intercept of second line
    findpk: whether to return peak (intersection of two lines); default False
    Outputs:
    --------
    if findpk: returns y values and the x location of the peak
    else: returns y values in array
    '''
    #y1 = s1*x + i1
    #y2 = s2*x + i2
    #find intersection point -- x value where y1 = y2
    peak = (i2 - i1)/(s1-s2)
    #use first line for x less than peak
    x1 = x[x<=peak]
    x2 = x[x>=peak]
    
    y1 = s1*x1+i1
    y2 = s2*x2+i2
    y = np.concatenate((y1,y2))
    if findpk: return y,peak
    else: return y


def triangfit(sog4,flareinds,ind,cut1=0,cut2=0,div=0,mult=False,off=False,rs=[],stcut=False,plot=True,chis=False,params=False):
    '''Fit triangle(s) to flare.
    Inputs:
    -------
    sog4
    flareinds
    ind: index of flare (within flareinds)
    cut1: number of data points to cut (or add if negative) from start
    cut2: number of data points to cut (or add if negative) from end
    div: index shift for split point for intitial fit (positive to move right)
    mult: boolean to use multiple initial fits (default False)
    rs: range of div to pass in
    off: boolean for rs to be different index cutoffs (more or less data used)
    stcut: boolean for rs to cut points from flare start rather than end
    plot: boolean for plotting; default True
    chis: boolean for returning chi-squared value(s); default to False
    params: boolean to return best fit triangle parameters; default False
    
    Outputs:
    --------
    if more and chis: peaks, chi-squared values, and reduced chi-squared values
    elif chis: best fit peak, chi-squared, and reduced chi-squared 
    elif more and params: best fit peaks, newx (x values spaced by a day), best fit parameters
    else: returns peak(s)
    
    To Do: allow for return of both chis and params'''
    #cutoff allows you to change the number of the points used in fit 
    st = flareinds[ind][0]+cut1
    end = flareinds[ind][1]-cut2
    x = sog4['MJD-50000'][st:end]
    newx = np.linspace(np.min(x),np.max(x)+1)
    y = sog4['I mag'][st:end]
    yerr = sog4['I mag err'][st:end]
    if plot: plt.scatter(x,y,color='cornflowerblue')
    split = int(div+(end-st)/2)
    #div moves where the cutoff is; positive div moves cutoff to right
    if mult: #mult can either be range of divisions or index cutoffs
        pks = []
        chi2 = []
        rchi2 = []
        for r in rs:
            if off: 
                if stcut: 
                    st = flareinds[ind][0]+r
                    #commented out changing split based on overwritten starts to separate variables
                    #split = int(div+(end-st)/2)
                else: 
                    end = flareinds[ind][1]-r
                    #split = int(div+(end-st)/2)
                #have to redefine what's ulimately fit
                #original flareinds x and y already scattered; showing any extra data with transparency
                x = sog4['MJD-50000'][st:end]
                y = sog4['I mag'][st:end]
                #to do: add errorbars
                yerr = sog4['I mag err'][st:end]
                #scatters again with last-defined x,y, so more points if negative rs used (and off=True)
                if plot: plt.scatter(x,y,alpha=.6,color='cornflowerblue')
            #split changed if off is False
            else: split = int(r+(end-st)/2)
            lin1 = np.polyfit(x[:split],y[:split], 1)
            lin2 = np.polyfit(x[split:],y[split:], 1)
            #split defined wrt beginning so has to be changed if stcut
            #want the split to still happen in the same place
            if stcut:
                 #subtract r to cancel out for r being added to start
                lin1 = np.polyfit(x[:split-r],y[:split-r], 1)
                #print(st+split-r) confirmed that the above line maintained the point of split
                lin2 = np.polyfit(x[split-r:],y[split-r:], 1)
            if plot: plt.plot(newx,triangle(newx,lin1[0],lin1[1],lin2[0],lin2[1]),color='navy',linestyle='dotted')
            #improve fit using curve_fit
            iv = [lin1[0],lin1[1],lin2[0],lin2[1]]  # for slope1, int1, slope2, int2
            #print(iv)
            bvs, covar = curve_fit(triangle, x, y, p0=iv)
            #using one point per day instead of x data
            b_fit,bpk = triangle(newx,bvs[0],bvs[1],bvs[2],bvs[3],findpk=True)
            b_fity = triangle(x,bvs[0],bvs[1],bvs[2],bvs[3])
            #print(bpk)
            pks.append(bpk)
            if plot: plt.plot(newx,b_fit,color='darkseagreen')
            #plt.legend()
            if chis:
                chi = np.sum((np.array(y)-np.array(b_fity))**2/(yerr**2))
                chi2.append(chi)
                #also add reduced chi squared since different numbers of points used if off
                rchi2.append(chi/(len(x)-4))
        if plot:
            plt.plot(newx,triangle(newx,lin1[0],lin1[1],lin2[0],lin2[1]),color='navy',linestyle='dotted',label='initial fit')
            plt.plot(newx,b_fit,color='darkseagreen',label='best fit')
            plt.ylim(np.max(y)+.02,np.min(y)-.02)
            plt.xlabel('MJD-50000')
            plt.ylabel('I mag')
            plt.legend()
        if chis: return pks,chi2,rchi2

        else: return pks
    else:
        split = int(div+(end-st)/2)
        lin1 = np.polyfit(x[:split],y[:split], 1)
        lin2 = np.polyfit(x[split:],y[split:], 1)
        #plt.scatter(x,y)
        if plot: plt.plot(newx,triangle(newx,lin1[0],lin1[1],lin2[0],lin2[1]),color='navy',label='initial fit',linestyle='dotted')
        #improve fit using curve_fit
        iv = [lin1[0],lin1[1],lin2[0],lin2[1]]  # for slope1, int1, slope2, int2
        bvs, covar = curve_fit(triangle, x, y, p0=iv)
        #fit for each day in range
        b_fit,bpk = triangle(newx,bvs[0],bvs[1],bvs[2],bvs[3],findpk=True)
        #fit for each data point
        b_fity = triangle(x,bvs[0],bvs[1],bvs[2],bvs[3])
        if ind == 2:
            print(bvs)
            print(bpk)
        if plot:
            plt.plot(newx,b_fit,color='darkseagreen',label='best fit')
            plt.legend()
            plt.ylim(np.max(y)+.02,np.min(y)-.02)
            plt.xlabel('MJD-50000')
            plt.ylabel('I mag')
        if chis:
            chi = np.sum((np.array(y)-np.array(b_fity))**2/(yerr**2))
            rchi = (chi/(len(x)-4)) #four triangle params
            return bpk, chi, rchi
        elif params: return bpk,newx,b_fit
        else: return bpk
        
def tri3D(sog4,flareinds,ind,cut1=np.arange(-2,2),cut2=np.arange(-2,2),div=np.arange(-2,2)):
    '''
    Inputs:
    -------
    sog4
    flareinds
    ind: index of flare in flareinds
    cut1: list or array of numbers of data points to try cutting (if positive) or adding to start of flare
    cut2: list or array of numbers of data points to try cutting (if positive) or adding to end of flare
    div: list or array of values to try adding or subtracting from default split index for initial fit
    
    Outputs:
    --------
    flarr: 3D array of best fit centers (where i,j,k are cut1,cut2, and div)
    flarr_r: reduced chi squared values for fits with each parameter combination
    '''
    flarr = np.zeros((len(cut1),len(cut2),len(div)))
    flarr_r = np.zeros((len(cut1),len(cut2),len(div)))
#     iind = int(-1*cut1/2)
    #print(iind)
#     jind = int(-1*cut2/2)
#     kind = int(-1*div/2)
    #for i in range(iind,np.abs(iind)):
    minc1 = np.min(cut1)
    minc2 = np.min(cut2)
    mindiv = np.min(div)
    for i in cut1:
        for j in cut2:
            for k in div:   
                cen,chi,rchi = triangfit(sog4,flareinds,ind,cut1=i,cut2=j,div=k,chis=True,plot=False)
                #set index zero point to lowest value
                flarr[i-minc1][j-minc2][k-mindiv] = cen
                flarr_r[i-minc1][j-minc2][k-mindiv] = rchi
    return flarr, flarr_r
def tripl(sog4,flareinds,ind,flarr,flarr_r,cut1=np.arange(-2,2),cut2=np.arange(-2,2)):
    '''
    Plots span of results from tri3D on scatter plot of flare, with line at fit with lowest reduced chi-squared.
    Inputs:
    -------
    sog4
    flareinds
    ind
    flarr,flarr_r: returns from tri3D
    cut1,cut2: arrays used for cut1 and cut2 in tri3D
    Output:
    -------
    center value with lowest reduced chi-squared
    '''
    st = flareinds[ind][0] #original start
    end = flareinds[ind][1] #original end
    #plots minimum used 
    maxc1 = np.max(cut1)
    maxc2 = np.max(cut2)
    minc1 = np.abs(np.min(cut1))
    minc2 = np.abs(np.min(cut2))
    plt.scatter(sog4['MJD-50000'][st+maxc1:end-maxc2],sog4['I mag'][st+maxc1:end-maxc2],color='navy',label='min data')
    #plots maximum used lighter
    plt.scatter(sog4['MJD-50000'][st-minc1:end+minc2],sog4['I mag'][st-minc1:end+minc2],color='navy',alpha=.4,label='max data')
    mincen = np.min(flarr)
    maxcen = np.max(flarr)
    plt.axvspan(mincen,maxcen,color='grey',alpha=.4)
    #add lowest reduced chi squared
    low = np.unravel_index(np.argmin(flarr_r, axis=None), flarr.shape)
    #center with lowest chi squared fit
    lowcen = flarr[low[0]][low[1]][low[2]]
    plt.axvline(lowcen)
    plt.ylim(np.max(sog4['I mag'][st:end])+.02,np.min(sog4['I mag'][st:end])-.02)
    return lowcen

def tp(sog4,flareinds,ind,cut1=np.arange(-2,2),cut2=np.arange(-2,2),div=np.arange(-2,2)):
    '''
    Executes both tri3D and tripl.
    '''
    flarr,flarr_r = tri3D(sog4,flareinds,ind,cut1=cut1,cut2=cut2,div=div)
    lowcen = tripl(sog4,flareinds,ind,flarr,flarr_r,cut1=cut1,cut2=cut2)
    return flarr,flarr_r,lowcen
       

def bstri(sog4,flareinds,ind,indiv=False,si=40,ei=130,num=1000,sp_off=0): #allows you to use flarinds or custom indices
    '''
    Bootstrap triangular fit.
    Inputs:
    -------
    sog4
    flareinds
    ind
    indiv: use individual index arguments (st,end) rather than ind, which then references flareinds (default False)
    st: used as start index if indiv True
    end: used as end index if indiv True
    num: number of bootstrap iterations (default 1000)
    sp_off: same as div; by how many indices to move initial split (default 0)
    
    Output:
    -------
    bspks: bootstrap centers
    '''
    bspks = []
    #use same split throughout
    if indiv:
        ind1 = si
        ind2 = ei
    else:
        ind1 = flareinds[ind][0]
        ind2 = flareinds[ind][1]
    t = sog4['MJD-50000']
    m = sog4['I mag']
#      st = np.array(t[ind1])
#     end = np.array(t[ind2])
    #index of split wrt beginning is beginning index plus optional argument plus the middle index of the flare
    split = ind1+sp_off+int((ind2-ind1)/2)
    #changing to use same initial conditions each time
    #print(split)
    lin1 = np.polyfit(sog4['MJD-50000'][ind1:split],sog4['I mag'][ind1:split], 1)
    lin2 = np.polyfit(sog4['MJD-50000'][split:ind2],sog4['I mag'][split:ind2], 1)
    iv = [lin1[0],lin1[1],lin2[0],lin2[1]]  # for slope1, int1, slope2, int2
    #find indices in flare -- not just st to end
    for i in range(num):
        bs = sk.resample(np.arange(ind1,ind2))
        #have to use full dataframe indexing for next two lines to work
        bst = np.array(sog4['MJD-50000'][bs])
        bsi = np.array(sog4['I mag'][bs])
        bsf = pd.DataFrame(columns=['t','i'])
        bsf['t'] = bst
        bsf['i'] = bsi
        bsf = bsf.sort_values(by='t')
        t,m = bsf['t'],bsf['i']
        #use arrays for fitting
        bvs, covar = curve_fit(triangle, t, m, p0=iv)
        b_fit,bpk = triangle(t,bvs[0],bvs[1],bvs[2],bvs[3],findpk=True)
        bspks.append(bpk)
    return bspks

def plbs(sog4,flareinds,bspks,ind,indiv=False,si=40,ei=132):
    '''
    Plot bootstrap span of results and hist.
    Inputs:
    -------
    sog4
    flareinds
    bspks: output of bstri (list of bootstrap results)
    ind: index in flareinds of flare
    indiv: use individual index arguments (st,end) rather than ind, which then references flareinds (default False)
    st: used as start index if indiv True
    end: used as end index if indiv True    
    '''
    if indiv:
        ind1 = si
        ind2 = ei
    else:
        ind1 = flareinds[ind][0]
        ind2 = flareinds[ind][1]
    fig,ax = plt.subplots(1,2,figsize=(12,5))
    imag = sog4['I mag'][ind1:ind2]
    ax[0].scatter(sog4['MJD-50000'][ind1:ind2],imag,color='#CF6275')
    ax[0].axvspan(np.min(bspks),np.max(bspks),color='grey',alpha=.3)
    im = ax[1].hist(bspks)
    ax[0].set_xlabel('MJD-50000',fontsize=14)
    ax[0].set_ylabel('I mag',fontsize=14)
    ax[1].set_xlabel('Center (MJD-50000)',fontsize=14)
    ax[0].set_ylim(np.max(imag)+.02,np.min(imag)-.02)
    return

#----------------------SINUSOIDAL FIT; updated----------------------
def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"
    Found this function online'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    #guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_freq = 1/175.
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

def peakplot(data,flareinds,mods,ind,plot=True):
    '''Find the flare center using sinusoidal model.
    Inputs:
    -------
    data: sog4
    flareinds
    mods: sine models
    ind: index of flare in flareinds
    plot: whether to plot flare and sine model (default True)
    Output:
    -------
    peak of sine model
    '''
    st = flareinds[ind][0]
    end = flareinds[ind][1]
    stdate = int(data['MJD-50000'][st:st+1])
    enddate = int(data['MJD-50000'][end-1:end])
    ff = np.arange(stdate,enddate,.1)
    pk = scipy.signal.find_peaks(-1*mods[ind]["fitfunc"](ff))[0]
    if plot:
        plt.scatter(data['MJD-50000'][st:end],data['I mag'][st:end],color='navy')
        plt.plot(np.linspace(stdate,enddate),mods[ind]["fitfunc"](np.linspace(stdate,enddate)))
        plt.ylim(np.max(data['I mag'][st:end])+.02,np.min(data['I mag'][st:end])-.02)
        plt.axvline(ff[pk])
    return np.float(ff[pk]) #returns center


