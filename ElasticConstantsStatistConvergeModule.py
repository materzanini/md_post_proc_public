__author__ = "Giuliana Materzanini & Tommaso Chiarotti"
__license__ = "MIT license, see LICENSE.txt file."
__version__ = "0.0.1"
__email__ = ""
"""
Utilities for calculating the elastic constants of method in doi:
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
bohr2a = 0.52917724899 
kBoltzAng10m9 = 1.380649e-2

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def read_from_file_evp(f_name):
    """"""
    file_evp = open(f_name,'r')
    every_step = []
    file_evp_lines = file_evp.readlines()
    nlinetot = len(file_evp_lines)
    nt = nlinetot
    for it in range(0,nlinetot):
        y = file_evp_lines[it].split()
        y = np.array(y,dtype=float)
        every_step.append(y)
    timeArray = np.array(every_step,dtype=float)
    file_evp.close()
    return timeArray

def read_cel_from_file_cel(f_name,nstep_cel):
    file_cel = open(f_name, 'r')
    file_cel_lines = file_cel.readlines()
    nlinetot = len(file_cel_lines)
    nt = int(len(file_cel_lines)/nstep_cel)
    celArray = np.zeros((nt,nstep_cel-1,3),dtype=float)
    celArrayA = np.zeros((nt,nstep_cel-1,3),dtype=float)
    for it in range(0,nt):
        every_nstep_cel = []
        for line in file_cel_lines[(nstep_cel*it)+1:nstep_cel*(it+1)]:
            y = line.split()
            y = np.array(y,dtype=float)
            every_nstep_cel.append(y)
        celArray[it,:,:] = np.array(every_nstep_cel,dtype=float)
        celArrayA[it,:,:] = celArray[it,:,:] * bohr2a
    file_cel.close()
    return celArrayA

def read_pos_from_file_pos(f_name,nstep_pos):
    """"""
    file_pos = open(f_name,'r')
    pos_every_step = []
    file_pos_lines = file_pos.readlines()
    nlinetot = len(file_pos_lines)
    nt = int(len(file_pos_lines)/nstep_pos)
    posArrayB = np.zeros((nt,nstep_pos-1,3),dtype=float)
    #
    for it in range(nt):
        pos_every_step = []
        for line in file_pos_lines[(nstep_pos*it)+1:nstep_pos*(it+1)]:
            y = line.split()
            y = np.array(y,dtype=float)
            pos_every_step.append(y)
        posArrayB[it,:,:] = np.array(pos_every_step,dtype=float)
    posArray = np.copy(posArrayB*bohr2a)
    file_pos.close()

    return posArray, nt

def cellVectAndAngles(celArray):
    nt = celArray.shape[0]
    a = np.zeros(int(nt),dtype=float)
    b = np.zeros(int(nt),dtype=float)
    c = np.zeros(int(nt),dtype=float)
    alpha = np.zeros(int(nt),dtype=float)
    beta = np.zeros(int(nt),dtype=float)
    gamma = np.zeros(int(nt),dtype=float)
    celArrayTransp = np.zeros((int(nt),3,3),dtype=float)
    for it in range(int(nt)):
        celArrayTransp[it,:,:] = celArray[it,:,:].transpose()

    volumeCell = np.zeros(int(nt),dtype=float)
    volAve = 0
    for it in range(int(nt)):
        a[it] = np.linalg.norm(celArray[it,:,0])
        b[it] = np.linalg.norm(celArray[it,:,1])
        c[it] = np.linalg.norm(celArray[it,:,2])
        #
        #
        alpha[it]=math.acos(np.vdot(celArray[it,:,1],celArray[it,:,2])/(b[it]*c[it]))
        beta[it] =math.acos(np.vdot(celArray[it,:,0],celArray[it,:,2])/(a[it]*c[it]))
        gamma[it]=math.acos(np.vdot(celArray[it,:,0],celArray[it,:,1])/(a[it]*b[it]))
        #
        #
        volumeCell[it] =np.linalg.det(celArray[it,:,:])
        #
        volAve = volAve + volumeCell[it] / int(nt)
    return nt,a,b,c,alpha,beta,gamma,volumeCell

def hTohCrystal(h):
    hCrystal = np.zeros(h.shape,dtype=float)
    nt = h.shape[0]
    for it in range(0,nt):
        a = np.linalg.norm(h[it,:,0])
        b = np.linalg.norm(h[it,:,1])
        c = np.linalg.norm(h[it,:,2])
        alpha = math.acos(np.vdot(h[it,:,1],h[it,:,2])/(b*c))
        beta = math.acos(np.vdot(h[it,:,0],h[it,:,2])/(a*c))
        gamma= math.acos(np.vdot(h[it,:,0],h[it,:,1])/(a*b))
        hCrystal[it,0,0] = a
        hCrystal[it,0,1] = b * math.cos(gamma)
        hCrystal[it,1,1] = b * math.sin(gamma)
        hCrystal[it,0,2] = c * math.cos(beta)
        hCrystal[it,1,2] = c * (math.cos(alpha) - math.cos(gamma) * math.cos(beta)) / math.sin(gamma)
        hCrystal[it,2,2] = math.sqrt(c**2 - (hCrystal[it,0,2]**2 + hCrystal[it,1,2]**2))
    return hCrystal

def htoVolume(h):
    Vol = np.linalg.det(h)
    return Vol

def hToEpsilon(h):
    hAve = np.mean(h,axis=0)
    volAve = np.linalg.det(hAve)
    hAveInv = np.linalg.inv(hAve)
    nt = h.shape[0]
    epsilon = np.zeros(h.shape,dtype=float)
    for it in range(0,nt):
        epsilon[it,:,:] = 0.5*((hAveInv.T)@(h[it,:,:].T)@h[it,:,:]@hAveInv - np.identity(h.shape[1]))
    return epsilon, volAve

def epsToepsVoigt(epsilon):
    epsVoigt = np.zeros((epsilon.shape[0],6),dtype=float)
    epsVoigt[:,0] = epsilon[:,0,0]
    epsVoigt[:,1] = epsilon[:,1,1]
    epsVoigt[:,2] = epsilon[:,2,2]
    epsVoigt[:,3] = epsilon[:,1,2] + epsilon[:,2,1]
    epsVoigt[:,4] = epsilon[:,0,2] + epsilon[:,2,0]
    epsVoigt[:,5] = epsilon[:,0,1] + epsilon[:,1,0]
    return epsVoigt

def block_mean_var(data,data_mean,n_block):
    """this function performs the block mean and the block variance of data"""
    N = data.shape[0]
    n_inblock = int(N/n_block)
    sigma2 = 0
    for iblock in range(n_block):
        mean_inblock = 0
        for datavalue in data[n_inblock*iblock:(iblock+1)*n_inblock]:
            mean_inblock = mean_inblock + datavalue/n_inblock
        sigma2 = sigma2 + (mean_inblock - data_mean)**2/(n_block)
    sigma2 = sigma2 / (n_block-1)
    delta_sigma2 = np.sqrt(2./(n_block-1)**3)*sigma2
    return sigma2,delta_sigma2

def epsVoigtToElasticConstants(epsV,volume,tempK):
    epsVAve = np.mean(epsV,axis=0)
    volAve = np.mean(volume)
    C = np.zeros((epsV.shape[1],epsV.shape[1]),dtype=float)
    S = np.zeros((epsV.shape[1],epsV.shape[1]),dtype=float)
    for i in range(6):
        for j in range(i,6):
            S[i,j] = np.mean(epsV[:,i]*epsV[:,j]) - epsVAve[i]*epsVAve[j]
    S = S + S.T - np.diag(S.diagonal())
    S = S*volAve/(kBoltzAng10m9*tempK)
    C = np.linalg.inv(S)
    return C, S

def elasticConstantsError(epsV,volume,S,C,tempK,nblock):
    volAve = np.mean(volume)
    sigma2Volume,deltasigma2Volume = block_mean_var(volume,volAve,nblock)
    epsVAve = np.zeros(epsV.shape[1],dtype=float)
    sigma2EpsV = np.zeros(epsV.shape[1],dtype=float)
    errsigma2EpsV = np.zeros(epsV.shape[1],dtype=float)
    sigma2EpsVEpsV = np.zeros((epsV.shape[1],epsV.shape[1]),dtype=float)
    errsigma2EpsVEpsV = np.zeros((epsV.shape[1],epsV.shape[1]),dtype=float)
    #
    for i in range(6):
        epsVAve[i] = np.mean(epsV[:,i])
        sigma2EpsV[i],errsigma2EpsV[i] = block_mean_var(epsV[:,i],epsVAve[i],nblock)
        for j in range(i,6):
            sigma2EpsVEpsV[i,j],errsigma2EpsVEpsV[i,j] = block_mean_var(epsV[:,i]*epsV[:,j],np.mean(epsV[:,i]*epsV[:,j]),nblock)
    sigma2S = np.zeros((epsV.shape[1],epsV.shape[1]),dtype=float)
    sigma2C = np.zeros((epsV.shape[1],epsV.shape[1]),dtype=float)
    for i in range(6):
        for j in range(i,6):
            sigma2S[i,j] = sigma2EpsVEpsV[i,j] + sigma2EpsV[i]*epsVAve[j]**2 + sigma2EpsV[j]*epsVAve[i]**2
    sigma2S = sigma2S + sigma2S.T - np.diag(sigma2S.diagonal())
    sigma2S = sigma2S * (volAve/(kBoltzAng10m9*tempK))**2
    sigma2S = sigma2S + (S/volAve)**2*sigma2Volume
    sigma2C = (C**2)@sigma2S@(C**2)
    return np.sqrt(sigma2C), np.sqrt(sigma2S), np.sqrt(sigma2Volume)

def ElasticConstantsToModuliAndErrors(C,sigma2CArray,S,sigma2SArray):
    BV = (C[0,0]+C[1,1]+C[2,2])/9 + 2*(C[0,1]+C[1,2]+C[0,2])/9
    GV = ((C[0,0]+C[1,1]+C[2,2]) - (C[0,1]+C[1,2]+C[0,2])+ 3 * (C[3,3]+C[4,4]+C[5,5]))/15
    #
    BR = 1/((S[0,0]+S[1,1]+S[2,2]) + 2*(S[0,1]+S[1,2]+S[0,2]))
    GR = 15/(4*(S[0,0]+S[1,1]+S[2,2]) - 4*(S[0,1]+S[1,2]+S[0,2])+3*(S[3,3]+S[4,4]+S[5,5]))
    #
    B = (BV+BR)/2
    G = (GV+GR)/2
    #
    E = 9*B*G/(3*B+G)
    nu = (3*B-2*G)/(2*(3*B+G))
    #
    sigma2BV = (sigma2CArray[0,0]+sigma2CArray[1,1]+sigma2CArray[2,2])/9**2 + 2**2*(sigma2CArray[0,1]+sigma2CArray[1,2]+sigma2CArray[0,2])/9**2
    sigma2GV = ((sigma2CArray[0,0]+sigma2CArray[1,1]+sigma2CArray[2,2]) + (sigma2CArray[0,1]+sigma2CArray[1,2]+sigma2CArray[0,2])+ 3**2 * (sigma2CArray[3,3]+sigma2CArray[4,4]+sigma2CArray[5,5]))/15**2
    #
    sigma2BR = ((sigma2SArray[0,0]+sigma2SArray[1,1]+sigma2SArray[2,2]) + 2**2*(sigma2SArray[0,1]+sigma2SArray[1,2]+sigma2SArray[0,2]))*BR**4
    sigma2GR = (4**2/15**2*(sigma2SArray[0,0]+sigma2SArray[1,1]+sigma2SArray[2,2])+4**2/15**2*(sigma2SArray[0,1]+sigma2SArray[1,2]+sigma2SArray[0,2])+1/5**2*(sigma2SArray[3,3]+sigma2SArray[4,4]+sigma2SArray[5,5]))*GR**4
    #
    sigma2B = (sigma2BV+sigma2BR)/4
    sigma2G = (sigma2GV+sigma2GR)/4
    #
    sigma2E = E**2 * (sigma2B/B**2 + sigma2G/G**2 + (3**2*sigma2B+sigma2G)/(3*B+G)**2)
    sigma2nu = nu**2 * ((3**2*sigma2B+2**2*sigma2G)/(3*B-2*G)**2 + (3**2*sigma2B+sigma2G)/(3*B+G)**2)
    return B, np.sqrt(sigma2B), G, np.sqrt(sigma2G), E, np.sqrt(sigma2E), nu, np.sqrt(sigma2nu)

def sigmaAndErr(nblock_step,tot_block,data,alat,temper,mater):
    """"""
    Ndata = data.shape[0]
    mean = np.mean(data)
    # tot_block=int(data.shape[0]/nblock_step)
    sigma2 = np.zeros(tot_block,dtype=float)
    delta_sigma2 = np.zeros(tot_block,dtype=float)
    arr_nblock = np.zeros(tot_block,dtype=float)
    data_in_block = np.zeros(tot_block,dtype=float)
    counter = 1
    sigma2[0] = float("inf")
    delta_sigma2[0] = 0
    for nblock in range(1,nblock_step*tot_block,nblock_step):
        sigma2[counter],delta_sigma2[counter] = block_mean_var(data,mean,nblock+1)
        arr_nblock[counter] = nblock + 1
        data_in_block[counter] = int(Ndata/(nblock+1))
        counter = counter + 1
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('$\sigma$ ($\AA^2$)', fontsize=14)
    ax1.set_xlabel('N. of data in block', fontsize=14)
    # ax1.set_title('Variance of correlated data as function of block number.')
    ax1.grid(b=True)
    ax1.xaxis.set_major_locator(MultipleLocator(10000))
    ax1.xaxis.set_minor_locator(MultipleLocator(1000))
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    # ax1.xaxis.grid(True, which='minor')
    sigma=np.sqrt(sigma2)
    delta_sigma= 0.5 * delta_sigma2/sigma
    ax1.errorbar(data_in_block,sigma,yerr=delta_sigma,linestyle='-',linewidth=0.5,label='$\sigma($' + alat + ')' + '\n' + mater + ', '+ '%4.0f'%temper + 'K')
    # ax1.legend(fontsize=16, loc=2)
    ax1.legend()

    return ax1,sigma,data_in_block


