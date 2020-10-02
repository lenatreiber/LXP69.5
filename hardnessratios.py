import numpy as np
from uncertainties import ufloat, unumpy
from uncertainties.umath import *
from astropy.io import fits
from astropy.table import Table
import pandas as pd
import matplotlib.pyplot as plt

def readevt(file,frame=False): 
    '''
    Reads evt file and returns arrays of times of events in seconds, MJD, PIs and OBSIDs, with 
    one value of each for each event
    Can also return as a DataFrame
    Prints object name and total number of events
    
    Input:
    ------
    file: evt file name, including path to directory
    frame: returns four arrays if False; returns pandas DataFrame if True
    
    Output:
    -------
    either four separate arrays or contained in DataFrame
    
    evt_data: full event table
    time: time of each event in seconds
    pi: energy of each event in units of 10eV
    obsid: OBSID of each event
    mjd: time of each event in MJD days; converts using start date from file header
    
    To Do:
    ------
    generalize in case other columns wanted
    make sure notebooks updated to new output
    add boolean for presence of obsid; this often makes it fail --> would then just 
    define how to bin
    '''
    hdu_list = fits.open(file, memmap=True)
    evt_data = Table(hdu_list[1].data)
    pi = np.array(evt_data['PI'])
    time = np.array(evt_data['TIME'])
    obsid = list(evt_data['OBSDIR'])
    wtsth = fits.getheader(file)
    objname = wtsth['OBJECT'] 
    print(objname)
    print('total events:',len(evt_data))
    mjd = timeconvert(time,wtsth)
    if frame:
        fr = pd.DataFrame(columns=['Time (s)','MJD','PI','OBSID'])
        fr['Time (s)'] = time
        fr['MJD'] = mjd
        fr['PI'] = pi
        fr['OBSID'] = obsid
        return fr
    else:
        return evt_data, time, pi, obsid, mjd
       
def timeconvert(time,header):
    '''
    Uses evt file header to convert times in seconds to MJD
    Called by readevt function
    
    Inputs:
    -------
    time: list or array of times of events in seconds
    header: header of evt file, already read by fits.getheader
    
    Outputs:
    --------
    timemjd: array of times of events in MJD (days)
    '''
    mjd = header['MJDREFI']+header['MJDREFF'] #start day
    t = np.array(time)
    timed = t/(3600*24) #time in days
    timemjd = timed+mjd #add start day
    return timemjd

def sepobs(obsid,time,mjd): 
    '''
    Separates events with different OBS IDs and calculates mean time in 
    seconds and MJD of each obsid dataset
    
    Inputs:
    -------
    obsid: list of obsids of all events; assumed to be in order
    time: list of times in seconds of each event
    mjd: list of times in MJD of each event
    
    Outputs:
    --------
    change: starting with 0, list of indices of events that are the first of 
            their obs id
    mt: list of mean time of each obs id in seconds
    mjt: list of mean time of each obs id in MJD
    '''
    change = [0]
    for i in range(1,len(obsid)):
        if obsid[i] != obsid[i-1]:
            change.append(i) #add index to change if obsid is different from previous
    mt = []
    mjt = []
    for c in range(1,len(change)):
        m = np.mean(time[change[c-1]:change[c]]) #mean time between start and end time
        mj = np.mean(mjd[change[c-1]:change[c]])
        mt.append(m)
        mjt.append(mj)
    mt.append(np.mean(time[change[-1]:])) #add mean times for final obs id
    mjt.append(np.mean(mjd[change[-1]:]))
    return change, mt, mjt

def ebin(pi,change,low,mid,high): 
    '''
    Inputs:
    -------
    pi: list or array of energies of events
    change: indices where obsid changes, starts with 0
    low: lower end of soft energy bin
    mid: cutoff between bins
    high: upper end of hard energy bin
    
    Outputs:
    --------
    sh: 2D array with each row an obsid, 
        0th column is number of soft X-rays
        1st column is number of hard X-rays
    hr: array of hardness ratio for each obsid
    rerr: errors of hardness ratios, 
        assuming Gaussian error on sh values
    others: array of photon energies excluded by 
        both bins
    '''
    sh = np.zeros((len(change),2)) #initalize array with length = number of observations
    others=0 #count of photons outside of both energy bins
    for c in range(1,len(change)):
        for i in range(change[c-1],change[c]):
            if pi[i]>=low and pi[i]<=mid: #soft bin includes cutoff
                sh[c-1][0]+=1 #adds one to soft bin of current obsid
            elif pi[i]>mid and pi[i]<=high:
                sh[c-1][1]+=1 #adds one to hard bin of current obs id
            else: others+=1
    for i in range(change[-1],len(pi)): #repeats for final obs id; could change to include last index in change
        if pi[i]>=low and pi[i]<=mid:
            sh[-1][0]+=1
        elif pi[i]>mid and pi[i]<=high:
            sh[-1][1]+=1   
        else: others+=1
    #compute hr and error
    hr=(sh[:,1]-sh[:,0])/(sh[:,1]+sh[:,0])
    se=np.sqrt(sh[:,0]) #assumes Gaussian error
    he=np.sqrt(sh[:,1])
    serr=unumpy.uarray(sh[:,0],se)
    herr=unumpy.uarray(sh[:,1],he)
    ratio=(herr-serr)/(herr+serr)
    rerr=np.zeros(len(change))
    for r in range(len(ratio)):
        rerr[r]=ratio[r].s #separates error
    return sh, hr, rerr, others

#plotting functions
def hrtime(time,hr,err,mjd=False,c='palevioletred'): #add label of ratio cutoffs
    plt.figure(figsize=(8,6))
    plt.errorbar(time,hr,color=c,yerr=err,linestyle='None',marker='o')
    if mjd:
        plt.xlabel('MJD (d)',fontsize=14) 
    else:
        plt.xlabel('Time (s)',fontsize=14)
    plt.ylabel('Hardness Ratio',fontsize=14)
    
    
    