import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from scipy.interpolate import interp1d
from matplotlib import colors
from datetime import datetime, timedelta
from scipy.optimize import minimize
from collections import Iterable

def find_index(values, vector):
    """ Given a set a values find the corresponding indicies 
     of nearest values in a vector
    
    Parameters:
    -----------

    values: 1-d array
        values for which you want to find the indices in vector

    vector: 1-d array

    Returns:
    --------

    1-d array of indices 
     
    """
    
    values = np.array([values]).flatten()
    minind = np.zeros(values.size)
    for ii in range(0,values.size):
        if np.isnan(values[ii]):
            continue
        else:
            diff = np.abs(vector-values[ii])
            minind[ii]=diff.argmin()
    return minind.astype(int)

def bin1d(x, y, step_x, bin_minmax=None, ppb=1, method='median', dropnans=False):
    """ 1-d binning
    
        Parameters:
        -----------

        x: 2-d array of size (n,m)
            2-d array with m columns where you want to bin the rows 

        y: 1-d array of size n
            1-d array that is used to perform the binning of x 
       
        step_x: float, pandas frequency alias (str)
            resolution, or distance between bin centers.

        bin_minmax: iterable with two values
            center of smallest and larges bin, optional

        ppb: int
            number of values per bin, if bin has too few values then it will
            be nan.

        method: str
            'median' returns the 25th, 50th and 75th percentiles for each bin 
            'mean' returns the mean and the standard deviation for each bin

        dropnans: bool
            If True it drops data rows that contain all NaN
            If False the all NaN are kept
    
        Returns:
        --------

        final_x: 1-d array (size k)
            Leading edges of bins    

        final_25,final_50,final_75: (2-d arrays, size: k,m)
            If method was median, these are the corresponding
            percentile values in the bins.  

        final_mean,final_std: (2-d arrays, size: k,m)
            If method was mean these are the mean and standard 
            deviation values in the bins. 
    
    """    

    # By default use the minimum and maximum values as the limits 
    if bin_minmax==None:
        bin_minmax=np.ones(2)
        bin_minmax[0]=np.nanmin(x)
        bin_minmax[1]=np.nanmax(x)

    # Check if the x values are floats or date objects
    if isinstance(x[0],float):
        temp_x = np.arange(bin_minmax[0], bin_minmax[1]+step_x/2., step_x)

    if isinstance(x[0],datetime):
        temp_x = pd.date_range(start=bin_minmax[0], end=bin_minmax[1], freq=step_x)

    data_x = temp_x[:-1]

    if len(y.shape)==1:
        y = y[np.newaxis].T

    data_25 = np.nan*np.ones((len(data_x),y.shape[1]))
    data_50 = np.nan*np.ones((len(data_x),y.shape[1]))
    data_75 = np.nan*np.ones((len(data_x),y.shape[1]))
    data_mean = np.nan*np.ones((len(data_x),y.shape[1]))
    data_std = np.nan*np.ones((len(data_x),y.shape[1]))

    findex = []
    
    for i in range(0,len(data_x)):
        y_block = y[((x>temp_x[i]) & (x<=temp_x[i+1])),:]

        # Turn infs into nans
        y_block[np.isinf(y_block)] = np.nan

        if len(y_block)>=ppb: # more than ppb amount of data was found in the bin
            if method=='median':
                data_25[i,:],data_50[i,:],data_75[i,:] = np.nanpercentile(y_block,[25,50,75],axis=0)
                findex.append(i)
            if method=='mean':
                data_mean[i,:] = np.nanmean(y_block,axis=0)
                data_std[i,:] = np.nanstd(y_block,axis=0)
                findex.append(i)
        else: # not enough data was found in the bin
            if dropnans==True:
                continue
            if method=='median':
                data_25[i,:],data_50[i,:],data_75[i,:]=np.nan,np.nan,np.nan
                findex.append(i)
            if method=='mean':
                data_mean[i,:] = np.nan
                data_std[i,:] = np.nan
                findex.append(i)

    
    final_x = data_x[findex]

    if method=='median':
        final_25 = data_25[findex,:]
        final_50 = data_50[findex,:]
        final_75 = data_75[findex,:]

        return final_x, final_25, final_50, final_75
    if method=='mean':
        final_mean = data_mean[findex,:]
        final_std = data_std[findex,:]

        return final_x, final_mean, final_std

def bin2d(z, x, y, step_x, step_y, minmax_x=None, minmax_y=None, ppb=1):
    """ 2-d binning

        Parameters:
        -----------
        
        z: 1-d array
            data corresponding to x and y coordinates
        
        x: 1-d array
            x coordinates

        y: 1-d array
            y coordinates

        step_x: float
            distance between x bin centers

        step_y: float
            distance between y bin centers

        minmax_x: iterable with two elements
            minmax_x[0] minimum value in x direction
            minmax_x[1] maximum value in x direction

        minmax_y: iterable with two elements
            minmax_y[0] minimum value in y direction 
            minmax_y[1] maximum value in y direction
            
        ppb: int
            minimum number of data points inside the 2-d bin 

        Returns:
        --------

        data_x: 1-d array
            bin centers in x direction

        data_y: 1-d array
            bin centers in y direction

        data_25, data_50, data_75: 2-d arrays
            2-d arrays with values corresponding to the bin centers
            rows correspond to x data
            columns correspond to the y data 

    """
    
    if (minmax_x==None) or (minmax_y==None):
        temp_x = np.arange(np.nanmin(x), np.nanmax(x) + step_x, step_x)
        temp_y = np.arange(np.nanmin(y), np.nanmax(y) + step_y, step_y)
    else:
        temp_x = np.arange(minmax_x[0], minmax_x[1] + step_x, step_x)
        temp_y = np.arange(minmax_y[0], minmax_y[1] + step_y, step_y)      

    data_x = ((temp_x[:-1] + temp_x[1:])/2.0)
    data_y = ((temp_y[:-1] + temp_y[1:])/2.0)

    data_25 = np.nan*np.ones((len(data_x),len(data_y)))
    data_50 = np.nan*np.ones((len(data_x),len(data_y)))
    data_75 = np.nan*np.ones((len(data_x),len(data_y)))

    for i in range(0,len(data_x)):
        for j in range(0,len(data_y)):
            z_block = z[(((x>temp_x[i]) & (x<=temp_x[i+1])) & ((y>temp_y[j]) & (y<=temp_y[j+1])))]
            
            z_block[np.isinf(z_block)] = np.nan

            if len(z_block)>=ppb: # data was found in the bin
                data_25[i,j],data_50[i,j],data_75[i,j] = np.nanpercentile(z_block,[25,50,75])
            else: # not enough data was foud in the bin
                continue

    return data_x, data_y, data_25, data_50, data_75

def rescale(x,mini,maxi,a,b):
    """ Linearly scale data in x between a and b """
    
    return ((b-a)*(x-mini)) / (maxi-mini) + a

def bearings2degrees(bearings):
    """ Convert bearings to regular degrees """

    degrees = np.zeros(bearings.shape)
    for ii in range(0,len(bearings)):
        if bearings[ii]<=90.0:
            degrees[ii]=90.0-bearings[ii];
        elif bearings[ii]<=180.0:
            degrees[ii]=(180-bearings[ii])+270.0;
        elif bearings[ii]<=270.0:
            degrees[ii]=(270.0-bearings[ii])+180.0;
        else:
            degrees[ii]=(360.0-bearings[ii])+90.0;
    return degrees

def plot_sumfile(handle,v,cbar=True,clim=(10,100000),cmap='jet',shading='flat',cbar_padding=0.05):    
    """ Plot UHEL's sum-formatted aerosol number-size distribution 
    
    Example:

    import numpy as np
    import matplotlib.pyplot as plt
    import aerosol_functions as af

    v = np.loadtxt("data.sum")

    fig,ax = plt.subplots()
    af.plot_sumfile(ax,v)
    plt.show()

    """
    
    time = v[1:,0]
    dp = v[0,2:]
    data = v[1:,2:]
    mesh_dp,mesh_time = np.meshgrid(dp,time)
    pcolorplot = handle.pcolormesh(mesh_time,mesh_dp,data,
                                   norm=colors.LogNorm(),
                                   linewidth=0,rasterized=True,cmap=cmap,shading=shading)
    handle.set_yscale('log')
    pcolorplot.set_clim(clim)
    pcolorplot.set_edgecolor('face')
    handle.autoscale(tight='true')

    if cbar:
        cbar = plt.colorbar(pcolorplot,ax=handle,
                            ticks=LogLocator(subs=range(10)),pad=cbar_padding)
        return pcolorplot,cbar
    else:
        return pcolorplot

def plot_sumfile2(handle,time,dp,data,clim=(10,100000),cmap='jet',shading='flat',cbar=True,cbar_padding=0.05):
    """ Plot UHEL's sum-formatted aerosol number-size distribution 
    
    Same as plot_sumfile() but time, dp and data
    are given as separate inputs.

    """
    

    mesh_dp,mesh_time = np.meshgrid(dp,time)
    pcolorplot = handle.pcolormesh(mesh_time,mesh_dp,data,
                                   norm=colors.LogNorm(),linewidth=0,
                                   rasterized=True,cmap=cmap,shading=shading)
    handle.set_yscale('log')
    pcolorplot.set_clim(clim)
    pcolorplot.set_edgecolor('face')
    handle.autoscale(tight='true')
    if cbar:
        cbar_ax = plt.colorbar(pcolorplot,ax=handle,pad=cbar_padding,
                            ticks=LogLocator(subs=range(10)))
        return pcolorplot,cbar_ax
    else:
        return pcolorplot

def dNdlogDp2dN(Dp,dNdlogDp):
    """ Convert from dNdlogDp to dN 
    
    Parameters:
    -----------

    Dp: 1-d array
        Geometric mean diameters for the size channels

    dNdlogDp: 2-d array
        The number size distribution with normalized concentrations

    Returns:
    --------

    dN: 2-d array
        The number size distribution with unnormalized concentrations 

    Assumptions:
    ------------
    - Dp are the geometric mean diameters of size channels.
      That means on a log scale logDp has to mark the middle point
      of the channel.
    - A size channel shares an edge with the size channel
      closest to it. This means there will not be overlap between 
      channels but if logDp are not evenly spaced, the size channels
      will have gaps between them.
    """

    logDp = np.log10(Dp) # Dp -> logDp
    dlogDp = np.zeros(logDp.shape) # initialize dlogDp array

    # The first and last logDp have only one
    # neighbour channel so their width have
    # only one option
    dlogDp[0] = logDp[1]-logDp[0]
    dlogDp[-1] = logDp[-1]-logDp[-2]

    # Calculate the rest of dlogDp.
    for i in range(1,len(logDp)-1):
        dlogDp[i]=np.min([logDp[i]-logDp[i-1],logDp[i+1]-logDp[i]])

    return dNdlogDp*dlogDp # dNdlogDp -> dN


def calc_CoagS(diam,time,Dp,dNdlogDp,temp,pres):
    """Calculate the coagualtion sink for particles size diam

    ARGUMENTS:
    ----------

    diam: float
        Particle diameter for which you want to calculate the CoagS
        unit: meters

    time: 1-d array, size n
        time in the data, unit: days
        unit : days

    Dp: 1-d array, size m
        diameter in the data, unit: meters
        unit : m

    dNdlogDp: 2-d array, size n,m
        dN/dlogDp matrix
        unit : cm-3

    temp: 1-d array, size n
        Ambient temperature corresponding to the data
        unit : K

    pres: 1-d array, size n
        Ambient pressure corresponding to the data
        unit : Pa

    RETURNS: 
    --------
    
    CoagS: 1-d array, size n
        Coagulation sink time series
        unit s-1

    """
    
    N = dNdlogDp2dN(Dp,dNdlogDp) # cm-3
    
    findex = np.argwhere(Dp>=diam).flatten()
    big_R = Dp[findex]/2.
    big_N = N[:,findex]
    k_B = 1.38064852e-23 # Boltzmann constant m2 kg s-2 K-1
    r0=diam/2.
    r1=r0
    dens=1000.
    CoagS=np.zeros(time.shape)
    for i in range(0,len(time)):
        lamda=(6.73e-8*temp[i]*(1+(110.4/temp[i])))/(296*pres[i]/101325.0*1.373)
        myy=(1.832e-5*(temp[i]**(1.5))*406.4)/(5093*(temp[i]+110.4))
        kn1=lamda/r1
        kn=lamda/big_R
        CC= 1.+(kn*(1.142+(0.558*np.exp((-.999)/kn))))
        CC1= 1. + (kn1*(1.142+(0.558*np.exp((-.999)/kn1))))
        D = (k_B*temp[i]*CC)/(6.*np.pi*myy*big_R)
        D1 = (k_B*temp[i]*CC1)/(6.*np.pi*myy*r1)
        M = 4./3.*np.pi*(big_R**3)*dens
        M1 = 4./3.*np.pi*(r1**3)*dens
        c= np.sqrt((8.*k_B*temp[i])/(np.pi*M))
        c1= np.sqrt((8.*k_B*temp[i])/(np.pi*M1))
        c12= np.sqrt((c**2)+(c1**2))
        r12= big_R+r1
        D12= D+D1 
        CCONT = 4.*np.pi*r12*D12
        CFR = np.pi*r12*r12*c12
        L=(8.*D)/(np.pi*c)
        L1=(8.*D1)/(np.pi*c1)
        SIG=(1./(3.*r12*L))*((r12+L)**3-(r12*r12+L*L)**1.5)-r12
        SIG1=(1./(3.*r12*L1))*((r12+L1)**3-(r12*r12+L1*L1)**1.5)-r12
        SIG12= np.sqrt((SIG**2)+(SIG1**2))
        KO=CCONT/((r12/(r12+SIG12))+(CCONT/CFR))
        CoagS[i] = np.nansum(KO*big_N[i,:]*1e6)
        if (r0==big_R[0]):
            CoagS[i] = 0.5*KO*big_N[i,0]*1e6+np.nansum(KO*big_N[i,1:]*1e6)
        else:
            CoagS[i] = np.nansum(KO*big_N[i,:]*1e6)
 
    return CoagS

def calc_CS(time, Dp, dNdlogDp, temp, pres):
    """ Calculate CS from aerosol size distribution

    ARGUMENTS:
    ----------
   
    time: 1-d array, size n
        unit: days

    Dp: 1-d array, size m
        unit m

    dNdlogDp: 2-d array, size n,m
      dN/dlogDp matrix, unit: cm-3

    temp: 1-d array, size n
        unit: K

    pres: 1-d array, size n
        unit: Pa


    RETURNS:
    --------

    CS: 1-d array, size n
        unit s-1

    """

    Mx=98.08
    Mair=28.965
    Pr=pres/101325.
    Dair=19.7
    Dx=51.96
    k = 8314.7
    R = Dp/2.0

    Temp = temp

    CS = np.nan * np.ones(time.shape)

    for i in range(len(time)):
        Dif = (0.001 * (Temp[i]**1.75)*np.sqrt( (1./Mair)+(1./Mx))) / (Pr[i]*(Dair**(1./3.)+Dx**(1./3.))**2)            

        lam=3.*(np.sqrt( (np.pi*Mx)/(8.*k*Temp[i]) )) * Dif *1e-4
            
        knud=lam/R;
            
        beta=(knud+1)/((0.377*knud)+1+(4/(3.*1.))*(knud**2)+(4/(3.*1))*knud)

        CS[i] = np.nansum((4.*np.pi*Dif)*dNdlogDp[i,:]*beta*R*1e2)

    return CS


def calc_concentration(v,dmin,dmax):
    """ Calculate particle number concentration from aerosol number-size distribution

    Args:
        v (2-D array): sum-formatted aerosol size distribution matrix (dN/dlogDp)
        dmin (float): lower limit for particle diameter
        dmax (float): upper limit for particle diameter
    Returns:
        conci (1-D array): number concentration between dmin and dmax
        time (1-D array): corresponding time points
    """

    dp = np.log10(v[0,2:])
    conc = v[1:,2:] 
    findex = np.argwhere((dp<=dmax)&(dp>=dmin)).flatten()
    dp = dp[findex]
    conc = data[:,findex]
    logdp_mid = np.log10(dp)
    logdp = (logdp_mid[:-1]+logdp_mid[1:])/2.0
    logdp = np.append(logdp,logdp_mid.max()+(logdp_mid.max()-logdp.max()))
    logdp = np.insert(logdp,0,logdp_mid.min()-(logdp.min()-logdp_mid.min()))
    dlogdp = np.diff(logdp)
    return np.nansum(conc*dlogdp,axis=1)

def calc_concentration2(dp,data,dmin,dmax):
    """ Calculate particle number concentration from aerosol number-size distribution
    
    Same as calc_concentration() but take the dp and data
    as separate inputs.
    """
    
    findex = np.argwhere((dp<=dmax)&(dp>=dmin)).flatten()
    dp = dp[findex]
    conc = data[:,findex]
    logdp_mid = np.log10(dp)
    logdp = (logdp_mid[:-1]+logdp_mid[1:])/2.0
    logdp = np.append(logdp,logdp_mid.max()+(logdp_mid.max()-logdp.max()))
    logdp = np.insert(logdp,0,logdp_mid.min()-(logdp.min()-logdp_mid.min()))
    dlogdp = np.diff(logdp)
    return np.nansum(conc*dlogdp,axis=1)

def cunn(Dp):
    """ Cunningham correction factor Makela et al. (1996) """

    return 1.+2.*64.5/Dp*(1.246+0.420*np.exp(-0.87*Dp/(2.*64.5))) 

def diam_to_mob(Dp): # [Dp] = nm
    """ Electrical mobility diameter [nm] -> electrical mobility [m2 s-1 V-1] """

    e = 1.60217662e-19 # Coulomb
    return (e*cunn(Dp))/(3.*np.pi*1.83245e-5*Dp*1e-9) # m2 s-1 V-1

def mob_to_diam(Zp):
    """ Electrical mobility [m2 s-1 V-1] -> electrical mobility diameter [m] """

    def minimize_this(Dp,Zp):
        return np.abs(diam_to_mob(Dp)-Zp)
    Dp0 = 0.0001 # initial guess in nm
    return minimize(minimize_this, Dp0, args=(Zp,), tol=1e-10).x[0]    

def datenum2datetime(matlab_datenum):
    """ Convert from matlab datenum to python datetime """

    python_datetimes = np.array([datetime.fromordinal(int(x)) + timedelta(days=x%1) - timedelta(days = 366) for x in matlab_datenum])

    return python_datetimes

def datetime2datenum(arg):
    """ Convert from python datetime to matlab datenum """

    if (isinstance(arg,Iterable)):
      out=[]
      for dt in arg:
        ord = dt.toordinal()
        mdn = dt + timedelta(days = 366)
        frac = (dt-datetime(dt.year,dt.month,dt.day,0,0,0)).seconds \
               / (24.0 * 60.0 * 60.0)
        out.append(mdn.toordinal() + frac)
      return np.array((out))
    else:
        dt = arg
        ord = dt.toordinal()
        mdn = dt + timedelta(days = 366)
        frac = (dt-datetime(dt.year,dt.month,dt.day,0,0,0)).seconds \
               / (24.0 * 60.0 * 60.0)
        return mdn.toordinal() + frac
