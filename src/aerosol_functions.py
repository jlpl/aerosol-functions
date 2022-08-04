"""
Collection of functions to analyze atmospheric 
aerosol data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from matplotlib import colors
from datetime import datetime, timedelta
from scipy.optimize import minimize
from collections.abc import Iterable

def plot_sumfile(
    ax,
    time,
    dp,
    dndlogdp,
    clim=(10,100000),
    cmap='jet',
    cscale='log',
    cbar=True,
    cbar_padding=0.05):    
    """ 
    Plot aerosol number-size distribution surface plot

    Parameters
    ----------

    ax : axes object
        axis on which to plot the data

    time : numpy 1d array, size n
        measurement times

    dp : numpy 1d array, size m 
        particle diameters

    dndlogdp : numpy 2d array, size (n,m)
        number-size distribution matrix

    clim : iterable with two numerical elements
        color limits

    cmap : str 
        colormap to be used

    cscale : str         
        type of color scale `log` or `linear`
        
    cbar : boolean
        plot colorbar `True` or `False`
    
    cbar_padding : float
        Control space between colorbar and axes
    
    Returns
    -------

    matplotlib.collections.QuadMesh
        
    Colorbar
        only if cbar is `True`

    """
    
    mesh_dp,mesh_time = np.meshgrid(dp,time)

    if cscale=='log':
        norm = colors.LogNorm()
        ticks = LogLocator(subs=range(10))
    elif cscale=='linear':
        norm = None
        ticks = None
    else:
        raise Exception('"cscale" must be "log" or "linear"') 

    pcolorplot = ax.pcolormesh(
        mesh_time,
        mesh_dp,
        dndlogdp,
        norm=norm,
        linewidth=0,
        rasterized=True,
        cmap=cmap)

    pcolorplot.set_clim(clim)
    ax.set_yscale('log')
    ax.autoscale(tight='true')
    ax.grid(which="both",c='k',ls=':',alpha=0.5)

    if cbar:
        cbar_ax = plt.colorbar(
            pcolorplot,
            ax=ax,
            pad=cbar_padding,
            ticks=ticks)
        return pcolorplot, cbar_ax
    else:
        return pcolorplot


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
