"""
Collection of functions to analyze atmospheric 
aerosol data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dts
from matplotlib.ticker import LogLocator
from matplotlib import colors
from datetime import datetime, timedelta
from scipy.optimize import minimize
from collections.abc import Iterable

def datenum2datetime(datenum):
    """
    Convert from matlab datenum to python datetime 

    Parameters
    ----------

    datenum : float or array of floats
        A serial date number representing the whole and 
        fractional number of days from 1-Jan-0000 to a 
        specific date (MATLAB datenum)

    Returns
    -------

    datetime or array of datetimes

    """

    if (isinstance(datenum,Iterable)):
        return np.array([datetime.fromordinal(int(x)) + timedelta(days=x%1) - timedelta(days = 366) for x in datenum])
    else:
        return datetime.fromordinal(int(datenum)) + timedelta(days=datenum%1) - timedelta(days = 366)

def datetime2datenum(dt):
    """ 
    Convert from python datetime to matlab datenum 

    Parameters
    ----------

    datetime or array of datetimes

    Returns
    -------

    float or array of floats
        A serial date number representing the whole and 
        fractional number of days from 1-Jan-0000 to a 
        specific date (MATLAB datenum)

    """

    if (isinstance(dt,Iterable)):
        out=[]
        for t in dt:
            ord = t.toordinal()
            mdn = t + timedelta(days = 366)
            frac = (t-datetime(t.year,t.month,t.day,0,0,0)).seconds \
                   / (24.0 * 60.0 * 60.0)
            out.append(mdn.toordinal() + frac)
        return np.array(out)
    else:
        ord = dt.toordinal()
        mdn = dt + timedelta(days = 366)
        frac = (dt-datetime(dt.year,dt.month,dt.day,0,0,0)).seconds \
               / (24.0 * 60.0 * 60.0)
        return mdn.toordinal() + frac

def bin1d(x, y, step_x, min_bin=None, max_bin=None, ppb=1):
    """ Utility function for binning data

    Parameters
    ----------

    x : 1-d array of size n
        1-d array along which the bins are calculated.

    y : 1-d array of size n or 2-d array of size (n,m)
        2-d array with m columns, the rows correspond to values in `x`

    step_x : float
        resolution, or distance between bin centers.  

    min_bin : float
        lower edge of minimum bin

    max_bin : float
        upper edge of maximum bin

    ppb : int
        number of values per bin, if bin has too few values then it will
        be `NaN`.

    Returns
    -------

    1-d array of size k
        bin centers

    1-d array of size k or 2-d array of size (k,m)
        median values in the bins

    1-d array of size k or 2-d array of size (k,m)
        25th percentile values in the bins

    1-d array of size k or 2-d array of size (k,m)
        75th percentile values in the bins

    """

    # By default use the minimum and maximum values as the limits
    if min_bin is None:
        min_bin=np.nanmin(x)
    if max_bin is None:
        max_bin=np.nanmax(x)

    temp_x = np.arange(min_bin, max_bin+step_x, step_x)

    data_x = (temp_x[:-1] + temp_x[1:])/2.

    if len(y.shape)==1:
        data_25 = np.nan*np.ones(len(data_x))
        data_50 = np.nan*np.ones(len(data_x))
        data_75 = np.nan*np.ones(len(data_x))

        for i in range(0,len(data_x)):
    
            y_block = y[((x>temp_x[i]) & (x<=temp_x[i+1]))]
            y_block[np.isinf(y_block)] = np.nan
    
            if len(y_block)>=ppb:
                data_25[i],data_50[i],data_75[i] = np.nanpercentile(y_block,[25,50,75],axis=0)
            else:
                continue

    else:
        data_25 = np.nan*np.ones((len(data_x),y.shape[1]))
        data_50 = np.nan*np.ones((len(data_x),y.shape[1]))
        data_75 = np.nan*np.ones((len(data_x),y.shape[1]))

        for i in range(0,len(data_x)):
    
            y_block = y[((x>temp_x[i]) & (x<=temp_x[i+1])),:]
            y_block[np.isinf(y_block)] = np.nan
    
            if len(y_block)>=ppb:
                data_25[i,:],data_50[i,:],data_75[i,:] = np.nanpercentile(y_block,[25,50,75],axis=0)
            else:
                continue

    return data_x, data_50, data_25, data_75

def generate_log_ticks():
    """
    Generate ticks and ticklabels for log axis
    """
    x=np.arange(1,10)
    y=np.arange(-10,-4).astype(float)
    log_ticks=[]
    log_tick_labels=[]
    for j in y:
        for i in x:
            log_ticks.append(np.log10(np.round(i*10**j,int(np.abs(j)))))
            if i==1:
                log_tick_labels.append("10$^{%d}$"%j)
            else:
                log_tick_labels.append('')

    log_ticks=np.array(log_ticks)
    return log_ticks,log_tick_labels

def plot_sumfile(
    time,
    dp,
    dndlogdp,
    ax=None,
    vmin=10,
    vmax=100000,
    cmap='turbo',
    interp='none',
    time_reso=2,
    time_formatter="%H:%M"):    
    """ 
    Plot aerosol particle number-size distribution surface plot

    Parameters
    ----------

    time : numpy 1d array, size n
        measurement times (MATLAB datenum)

    dp : numpy 1d array, size m 
        particle diameters

    dndlogdp : numpy 2d array, size (n,m)
        number-size distribution matrix

    ax : axes object
        axis on which to plot the data
        if `None` the axis are created.

    vmin : float or int
        color scale lower limit

    vmax : float or int
        color scale upper limit

    clim : iterable with two numerical elements
        color limits

    cmap : `str`
        colormap to be used

    interp : `str`
        interpolation method passed to imshow, default `'none'`

    time_reso : `int`
        Resolution on the time axis given in hours

    time_formatter : `str`
        Define the format of time ticklabels

    """

    if ax is None:
        fig,handle = plt.subplots()
    else:
        handle=ax

    log_ticks,log_tick_labels = generate_log_ticks()

    norm = colors.LogNorm(vmin=vmin,vmax=vmax)
    color_ticks = LogLocator(subs=range(10))

    handle.set_yticks(log_ticks)
    handle.set_yticklabels(log_tick_labels)

    t1=dts.date2num(datenum2datetime(time.min()))
    t2=dts.date2num(datenum2datetime(time.max()))

    img = handle.imshow(
        np.flipud(dndlogdp.T),
        origin="upper",
        aspect="auto",
        interpolation=interp,
        cmap=cmap,
        norm=norm,
        extent=(t1,t2,np.log10(dp.min()),np.log10(dp.max()))
    )

    handle.xaxis.set_major_locator(dts.HourLocator(interval=time_reso))
    handle.xaxis.set_major_formatter(dts.DateFormatter(time_formatter))
    plt.setp(handle.get_xticklabels(),rotation=80)

    box = handle.get_position()
    c_handle = plt.axes([box.x0*1.025 + box.width * 1.025, box.y0, 0.01, box.height])
    cbar = plt.colorbar(img,cax=c_handle,ticks=color_ticks)

    handle.set_ylabel('Dp, [m]')
    handle.set_xlabel('Time')
    cbar.set_label('dN/dlogDp, [cm-3]')

    if ax is None:
        plt.show()

def dndlogdp2dn(dp,dndlogdp):
    """    
    Convert from normalized number concentrations to
    unnormalized number concentrations assuming that 
    the size channels have common edges.

    Parameters
    ----------

    dp : numpy 1d array
        Geometric mean diameters for the size channels

    dndlogdp : numpy 2d array
        Number size distribution with normalized concentrations
        i.e. dN/dlogDp

    Returns
    -------

    2-d array
        The number size distribution with unnormalized concentrations 
        i.e. dN

    """

    logdp_mid = np.log10(dp)
    logdp = (logdp_mid[:-1]+logdp_mid[1:])/2.0
    logdp = np.append(logdp,logdp_mid.max()+(logdp_mid.max()-logdp.max()))
    logdp = np.insert(logdp,0,logdp_mid.min()-(logdp.min()-logdp_mid.min()))
    dlogdp = np.diff(logdp)
    return dndlogdp*dlogdp


def air_viscosity(temp):
    """ 
    Calculate air viscosity
    using Enskog-Chapman theory

    Parameters
    ----------

    temp : float or array
        air temperature, unit: K  

    Returns
    -------

    float or array
        viscosity of air, unit: m2 s-1  

    """

    nyy_ref=18.203e-6
    S=110.4
    temp_ref=293.15
    return nyy_ref*((temp_ref+S)/(temp+S))*((temp/temp_ref)**(3./2.))

def mean_free_path(temp,pres):
    """ 
    Calculate mean free path in air

    Parameters
    ----------

    temp : float
        air temperature, unit: K  

    pres : float
        air pressure, unit: Pa

    Returns
    -------

    float
        mean free path in air, unit: m  

    """

    R=8.3143
    Mair=0.02897
    mu=air_viscosity(temp)
    return (2.*mu)/(pres*(8.*Mair/(np.pi*R*temp))**(1./2.))

def slipcorr(dp,temp,pres):
    """
    Slip correction factor in air 

    Parameters
    ----------

    dp : float or numpy array
        particle diameter, unit: m 

    temp : float
        air temperature, unit: K 

    pres : float
        air pressure, unit: Pa

    Returns
    -------

    float or numpy array
        Cunningham slip correction factor for each particle diameter, 
        unit: dimensionless        

    """
   
    l = mean_free_path(temp,pres)
    return 1.+((2.*l)/dp)*(1.257+0.4*np.exp(-(1.1*dp)/(2.*l)))

def particle_diffusivity(dp,temp,pres):
    """ 
    Particle brownian diffusivity in air 

    Parameters
    ----------

    dp : float or array
        particle diameter, unit: m 

    temp : float
        air temperature, unit: K 

    pres : float
        air pressure, unit: Pa

    Returns
    -------

    float or array
        Brownian diffusivity in air for particles of size dp,
        unit: m2 s-1

    """

    k=1.381e-23
    cc=slipcorr(dp,temp,pres)
    mu=air_viscosity(temp)

    return (k*temp*cc)/(3.*np.pi*mu*dp)

def particle_thermal_speed(dp,temp):
    """
    Particle thermal speed 

    Parameters
    ----------

    dp : float or array
        particle diameter, unit: m 

    temp : float
        air temperature, unit: K 

    Returns
    -------

    float or array
        Particle thermal speed for each dp, unit: m s-1

    """

    k=1.381e-23
    rho_p=1000.0
    mp=rho_p*(1./6.)*np.pi*dp**3.
    
    return ((8.*k*temp)/(np.pi*mp))**(1./2.)

def particle_mean_free_path(dp,temp,pres):
    """ 
    Particle mean free path in air 

    Parameters
    ----------

    dp : float or array
        particle diameter, unit: m 

    temp : float
        air temperature, unit: K 

    pres : float
        air pressure, unit: Pa

    Returns
    -------

    float or array
        Particle mean free path for each dp, unit: m

    """

    D=particle_diffusivity(dp,temp,pres)
    c=particle_thermal_speed(dp,temp)

    return (8.*D)/(np.pi*c)

def coagulation_coef(dp1,dp2,temp,pres):
    """ 
    Calculate Brownian coagulation coefficient (Fuchs)

    Parameters
    ----------

    dp1 : float
        first particle diameter, unit: m 

    dp2 : float
        second particle diameter, unit: m 

    temp : float
        air temperature, unit: K 

    pres : float
        air pressure, unit: Pa

    Returns
    -------

    float or array
        Brownian coagulation coefficient (Fuchs), unit: m3 s-1

    """

    def particle_g(dp,temp,pres):
        l = particle_mean_free_path(dp,temp,pres)    
        return 1./(3.*dp*l)*((dp+l)**3.-(dp**2.+l**2.)**(3./2.))-dp

    D1 = particle_diffusivity(dp1,temp,pres)
    D2 = particle_diffusivity(dp2,temp,pres)
    g1 = particle_g(dp1,temp,pres)
    g2 = particle_g(dp2,temp,pres)
    c1 = particle_thermal_speed(dp1,temp)
    c2 = particle_thermal_speed(dp2,temp)
    
    return 2.*np.pi*(D1+D2)*(dp1+dp2) \
           * ( (dp1+dp2)/(dp1+dp2+2.*(g1**2.+g2**2.)**0.5) + \
           +   (8.*(D1+D2))/((c1**2.+c2**2.)**0.5*(dp1+dp2)) )

def calc_coags(Dp,dp,dndlogdp,temp,pres):
    """ 
    Calculate coagulation sink

    Kulmala et al (2012): doi:10.1038/nprot.2012.091 

    Parameters
    ----------

    Dp : float
        Particle diameter for which you want to calculate the CoagS, 
        unit: m

    dp : numpy 1d array, size m
        diameter in the data, unit: meters,
        unit: m

    dndlogdp : numpy 2d array, size (n,m)
        dN/dlogDp matrix,
        unit: cm-3

    temp : float or numpy 1d array of size n
        Ambient temperature corresponding to the data,
        unit: K

    pres : float or numpy 1d array of size n
        Ambient pressure corresponding to the data,
        unit: Pa

    Returns
    -------
    
    numpy 1d array, size n
        Coagulation sink time series,
        unit: s-1

    """

    n = dndlogdp.shape[0]

    if not isinstance(temp,Iterable):
        temp = temp*np.ones(n)

    if not isinstance(pres,Iterable):
        pres = pres*np.ones(n)

    dn = dndlogdp2dn(dp,dndlogdp)
    dp = dp[dp>=Dp]
    dn = dn[:,dp>=Dp]

    coags = np.nan*np.ones(n)

    for i in range(n):
        # multiply by 1e6 to make [K] = cm3 s-1
        coags[i] = np.nansum(1e6*coagulation_coef(Dp,dp,temp[i],pres[i])*dn[i,:])
                
    return coags
    
def diam2mob(dp,temp,pres,ne):
    """ 
    Convert electrical mobility diameter to electrical mobility in air

    Parameters
    ----------

    dp : float or numpy 1d array
        particle diameter(s),
        unit : m

    temp : float
        ambient temperature, 
        unit: K

    pres : float
        ambient pressure, 
        unit: Pa

    ne : int
        number of charges on the aerosol particle

    Returns
    -------

    float or numpy 1d array
        particle electrical mobility or mobilities, 
        unit: m2 s-1 V-1

    """

    e = 1.60217662e-19
    cc = slipcorr(dp,temp,pres)
    mu = air_viscosity(temp)

    Zp = (ne*e*cc)/(3.*np.pi*mu*dp)

    return Zp

def mob2diam(Zp,temp,pres,ne):
    """
    Convert electrical mobility to electrical mobility diameter in air

    Parameters
    ----------

    Zp : float or numpy 1d array
        particle electrical mobility or mobilities, 
        unit: m2 s-1 V-1

    temp : float
        ambient temperature, 
        unit: K

    pres : float
        ambient pressure, 
        unit: Pa

    ne : integer
        number of charges on the aerosol particle

    Returns
    -------

    float or numpy 1d array
        particle diameter(s), unit: m
    
    """

    def minimize_this(dp,Z):
        return np.abs(diam2mob(dp,temp,pres,ne)-Z)

    dp0 = 0.0001

    result = minimize(minimize_this, dp0, args=(Zp,), tol=1e-20, method='Nelder-Mead').x[0]    

    return result

def binary_diffusivity(temp,pres,Ma,Mb,Va,Vb):
    """ 
    Binary diffusivity in a mixture of gases a and b

    Fuller et al. (1966): https://doi.org/10.1021/ie50677a007 

    Parameters
    ----------

    temp : float
        temperature, 
        unit: K

    pres : float
        pressure, 
        unit: Pa

    Ma : float
        relative molecular mass of gas a, 
        unit: dimensionless

    Mb : float
        relative molecular mass of gas b, 
        unit: dimensionless

    Va : float
        diffusion volume of gas a, 
        unit: dimensionless

    Vb : float
        diffusion volume of gas b, 
        unit: dimensionless

    Returns
    -------

    float
        binary diffusivity, 
        unit: m2 s-1

    """
    
    diffusivity = (1.013e-2*(temp**1.75)*np.sqrt((1./Ma)+(1./Mb)))/(pres*(Va**(1./3.)+Vb**(1./3.))**2)
    return diffusivity


def beta(dp,temp,pres,diffusivity,molar_mass):
    """ 
    Calculate Fuchs Sutugin correction factor 

    Sutugin et al. (1971): https://doi.org/10.1016/0021-8502(71)90061-9

    Parameters
    ----------

    dp : float or numpy 1d array
        aerosol particle diameter(s), 
        unit: m

    temp : float
        temperature, 
        unit: K

    pres : float
        pressure,
        unit: Pa

    diffusivity : float
        diffusivity of the gas that is condensing, 
        unit: m2/s

    molar_mass : float
        molar mass of the condensing gas, 
        unit: g/mol

    Returns
    -------

    float or 1-d numpy array
        Fuchs Sutugin correction factor for each particle diameter, 
        unit: m2/s

    """

    R = 8.314 
    l = 3.*diffusivity/((8.*R*temp)/(np.pi*molar_mass*0.001))**0.5
    knud = 2.*l/dp
    
    return (1. + knud)/(1. + 1.677*knud + 1.333*knud**2)

def calc_cs(dp,dndlogdp,temp,pres):
    """
    Calculate condensation sink, assuming that the condensing gas is sulfuric acid in air
    with aerosol particles.

    Kulmala et al (2012): doi:10.1038/nprot.2012.091 

    Parameters
    ----------

    dp : numpy 1d array, size m
        diameter in the data, unit: m

    dndlogdp : numpy 2d array, size (n,m)
        dN/dlogDp matrix, unit: cm-3

    temp : numpy 1d array, size n
        Ambient temperature corresponding to the data, unit: K

    pres : numpy 1d array, size n
        Ambient pressure corresponding to the data, unit: Pa

    Returns
    -------
    
    numpy 1d array, size n
        condensation sink time series, unit: s-1

    """

    n = dndlogdp.shape[0]

    if not isinstance(temp,Iterable):
        temp=temp*np.ones(n)

    if not isinstance(pres,Iterable):
        pres=pres*np.ones(n)

    M_h2so4 = 98.08   
    M_air = 28.965    
    V_air = 19.7      
    V_h2so4 = 51.96  

    dn = dndlogdp2dn(dp,dndlogdp)
    cs = np.nan*np.ones(n)

    for i in range(n):
        diffusivity = binary_diffusivity(temp[i],pres[i],M_h2so4,M_air,V_h2so4,V_air)
        b = beta(dp,temp[i],pres[i],diffusivity,M_h2so4)

        cs[i] = (4.*np.pi*diffusivity)*np.nansum(1e6*dn[i,:]*b*dp)

    return cs

def calc_conc(dp,dndlogdp,dmin,dmax):
    """
    Calculate particle number concentration from aerosol 
    number-size distribution

    Parameters
    ----------

    dp : numpy 1d array, size m
        diameter in the data, unit: m

    dndlogdp : numpy 2d array, size (n,m)
        dN/dlogDp matrix, unit: cm-3

    dmin : float
        Size range lower diameter, unit: m

    dmax : float
        Size range upper diameter, unit: m

    Returns
    -------
    
    numpy 1d array, size n
        Number concentration in the given size range, unit: cm-3

    """
    
    findex = np.argwhere((dp<=dmax)&(dp>=dmin)).flatten()
    dp = dp[findex]
    dndlogdp = dndlogdp[:,findex]
    logdp_mid = np.log10(dp)
    logdp = (logdp_mid[:-1]+logdp_mid[1:])/2.0
    logdp = np.append(logdp,logdp_mid.max()+(logdp_mid.max()-logdp.max()))
    logdp = np.insert(logdp,0,logdp_mid.min()-(logdp.min()-logdp_mid.min()))
    dlogdp = np.diff(logdp)
    return np.nansum(dndlogdp*dlogdp,axis=1)

def calc_formation_rate(time,dp1,dp2,conc,coags,gr):
    """
    Calculate particle formation rate

    Kulmala et al (2012): doi:10.1038/nprot.2012.091

    Parameters
    ----------

    time : numpy 1d array
        time associated with the measurements

    dp1 : float
        Lower diameter of the size range, unit: m

    dp2 : float
        Upper diameter of the size range, unit: m

    conc : numpy 1d array
        Particle number concentration in the size range dp1...dp2, unit: cm-3

    coags : numpy 1d array
        Coagulation sink for particles in the size range dp1...dp2. Usually approximated as coagulation sink for particle size dp1, unit: s-1

    gr : float
        Growth rate for particles out of the size range dp1...dp2, unit: nm h-1

    Returns
    -------

    numpy 1d array
        particle formation rate for diameter dp1, unit: cm3 s-1

    """

    conc_term = np.diff(conc)/np.diff(time*1.157e5)
    sink_term = (coags[1:] + coags[:-1])/2. * (conc[1:] + conc[:-1])/2.
    gr_term = (2.778e-13*gr)/(dp2-dp1) * (conc[1:] + conc[:-1])/2.
    formation_rate = conc_term + sink_term + gr_term

    return formation_rate

def calc_ion_formation_rate(
    time,
    dp1,
    dp2,
    conc_pos,
    conc_neg,
    conc_pos_small,
    conc_neg_small,
    conc,
    coags,
    gr):
    """ 
    Calculate ion formation rate

    Kulmala et al (2012): doi:10.1038/nprot.2012.091

    Parameters
    ----------

    time : numpy 1d array
        Time associated with the measurements, unit: days  

    dp1 : float
        Lower diameter of the size range, unit: m

    dp2 : float
        Upper diameter of the size range, unit: m

    conc_pos : numpy 1d array
        Positive ion number concentration in the size range dp1...dp2, unit: cm-3

    conc_neg : numpy 1d array
        Negative ion number concentration in the size range dp1...dp2, unit: cm-3

    conc_pos_small : numpy 1d array
        Positive ion number concentration for ions smaller than dp1, unit: cm-3

    conc_neg_small : numpy 1d array
        Negative ion number concentration for ions smaller than dp1, unit: cm-3

    conc : numpy 1d array
        Particle number concentration in the size range dp1...dp2, unit: cm-3

    coags : numpy 1d array
        Coagulation sink for particles in the size range dp1...dp2.
        Usually approximated as coagulation sink for particle size dp1, 
        unit: s-1

    gr : float
        Growth rate for particles out of the size range dp1...dp2, unit: nm h-1

    Returns
    -------

    numpy 1d array
        Positive ion formation rate for diameter dp1, unit : cm3 s-1

    numpy 1d array
        Negative ion formation rate for diameter dp1, unit: cm3 s-1

    """

    alpha = 1.6e-6 # cm3 s-1
    Xi = 0.01e-6 # cm3 s-1

    coags = (coags[1:] + coags[:-1])/2.
    conc_pos = (conc_pos[1:] + conc_pos[:-1])/2.
    conc_neg = (conc_neg[1:] + conc_neg[:-1])/2.
    conc_pos_small = (conc_pos_small[1:] + conc_pos_small[:-1])/2.
    conc_neg_small = (conc_neg_small[1:] + conc_neg_small[:-1])/2.
    conc = (conc[1:] + conc[:-1])/2.

    pos_conc_term = np.diff(conc_pos)/np.diff(time*1.157e5)
    pos_sink_term = coags * conc_pos
    pos_gr_term = (2.778e-13*gr)/(dp2-dp1) * conc_pos
    pos_recombination_term = alpha * conc_pos * conc_neg_small
    pos_charging_term = Xi * conc * conc_pos_small
    pos_formation_rate = pos_conc_term + pos_sink_term + pos_gr_term + pos_recombination_term - pos_charging_term

    neg_conc_term = np.diff(conc_neg)/np.diff(time*1.157e5)
    neg_sink_term = coags * conc_neg
    neg_gr_term = (2.778e-13*gr)/(dp2-dp1) * conc_neg
    neg_recombination_term = alpha * conc_neg * conc_pos_small
    neg_charging_term = Xi * conc * conc_neg_small
    neg_formation_rate = neg_conc_term + neg_sink_term + neg_gr_term + neg_recombination_term - neg_charging_term

    return pos_formation_rate, neg_formation_rate
