"""
Aerosol number-size distribution is assumed 
to be a pandas DataFrame where

index: 
    time, pandas.DatetimeIndex
columns: 
    size bin diameters in meters, float
values: 
    normalized concentration dN/dlogDp in cm-3, float



"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dts
from matplotlib import colors
from matplotlib.pyplot import cm
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy.interpolate import interp1d

def air_density(temp,pres):
    """
    Calculate air density

    Parameters
    ----------

    temp : float or array   
        absolute temperature (K) 
    pres : float or array
        absolute pressure (Pa)
 
    Returns
    -------

    float or array
        air density (kg/m3)
    """
    return pres/(287.0500676*temp)

def datenum2datetime(datenum):
    """
    Convert from matlab datenum to python datetime 

    Parameters
    ----------

    datenum : float or int
        A serial date number representing the whole and 
        fractional number of days from 1-Jan-0000 to a 
        specific date (MATLAB datenum)

    Returns
    -------

    pandas.Timestamp

    """

    return (datetime.fromordinal(int(datenum)) + 
        timedelta(days=datenum%1) - timedelta(days = 366))

def datetime2datenum(dt):
    """ 
    Convert from python datetime to matlab datenum 

    Parameters
    ----------

    dt : datetime object

    Returns
    -------

    float
        A serial date number representing the whole and 
        fractional number of days from 1-Jan-0000 to a 
        specific date (MATLAB datenum)

    """

    ord = dt.toordinal()
    mdn = dt + timedelta(days = 366)
    frac = (dt-datetime(dt.year,dt.month,dt.day,0,0,0)).seconds / (24.0 * 60.0 * 60.0)
    return mdn.toordinal() + frac


def calc_bin_edges(dp):
    """
    Calculate bin edges for log-spaced bin centers
    
    Parameters
    ----------
    
    dp : numpy.array (n,)
        bin center diameters

    Returns
    -------

    numpy.array (n+1,)
        log bin edges

    """

    logdp_mid = np.log10(dp)
    logdp = (logdp_mid[:-1]+logdp_mid[1:])/2.0
    logdp = np.append(logdp,logdp_mid.max()+(logdp_mid.max()-logdp.max()))
    logdp = np.insert(logdp,0,logdp_mid.min()-(logdp.min()-logdp_mid.min()))
    
    return logdp

def dndlogdp2dn(df):
    """    
    Convert from normalized number concentrations to
    unnormalized number concentrations.

    Parameters
    ----------

    df : pandas.DataFrame
        Aerosol number-size distribution (dN/dlogDp)

    Returns
    -------

    pandas.DataFrame
        Aerosol number size distribution (dN)

    """
    
    dp = df.columns.values.astype(float)
    logdp = calc_bin_edges(dp)
    dlogdp = np.diff(logdp)

    return df*dlogdp

def air_viscosity(temp):
    """ 
    Calculate air viscosity using Enskog-Chapman theory

    Parameters
    ----------

    temp : float or numpy.array
        air temperature, unit: K  

    Returns
    -------

    float or numpy.array
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

    temp : float or numpy.array
        air temperature, unit: K  
    pres : float or numpy.array
        air pressure, unit: Pa

    Returns
    -------

    float or numpy.array
        mean free path in air, unit: m

    """

    R=8.3143
    Mair=0.02897
    mu=air_viscosity(temp)
    return (mu/pres)*((np.pi*R*temp)/(2.*Mair))**0.5

def slipcorr(dp,temp,pres):
    """
    Slip correction factor in air 

    Parameters
    ----------

    dp : float or numpy array (m,)
        particle diameter, unit m 
    temp : float or numpy.array (n,1)
        air temperature, unit K 
    pres : float or numpy.array (n,1)
        air pressure, unit Pa

    Returns
    -------

    float or numpy.array (m,) or (n,m)
        Cunningham slip correction factor for each particle diameter,
        if temperature and pressure and arrays then for each particle 
        diameter at different pressure/temperature values.
        unit dimensionless


    Notes
    -----

    Correction is done according to Mäkelä et al. (1996)

    """
   
    l = mean_free_path(temp,pres)
    return 1.+((2.*l)/dp)*(1.246+0.420*np.exp(-(0.87*dp)/(2.*l)))

def particle_diffusivity(dp,temp,pres):
    """ 
    Particle brownian diffusivity in air 

    Parameters
    ----------

    dp : float or numpy.array (m,) 
        particle diameter, unit: m 
    temp : float or numpy.array (n,1)
        air temperature, unit: K 
    pres : float or numpy.array (n,1)
        air pressure, unit: Pa

    Returns
    -------

    float or numpy.array (m,) or (n,m)
        Brownian diffusivity in air for particles of size dp,
        and at each temperature/pressure value
        unit m2 s-1

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

    dp : float or numpy.array (m,)
        particle diameter, unit: m 
    temp : float or numpy.array (n,1)
        air temperature, unit: K 

    Returns
    -------

    float or numpy.array (m,) or (n,m)
        Particle thermal speed for each dp at each temperature 
        point, unit: m s-1

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

    dp : float or numpy.array (m,)
        particle diameter, unit: m 
    temp : float or numpy.array (n,1)
        air temperature, unit: K 
    pres : float or numpy.array (n,1)
        air pressure, unit: Pa

    Returns
    -------

    float or numpy.array (m,) or (n,m)
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

    dp1 : float or numpy.array (m,)
        first particle diameter, unit: m 
    dp2 : float or numpy.array (m,)
        second particle diameter, unit: m 
    temp : float or numpy.array (n,1)
        air temperature, unit: K 
    pres : float or numpy.array (n,1)
        air pressure, unit: Pa

    Returns
    -------

    float or numpy.array
        Brownian coagulation coefficient (Fuchs), 
        
        for example if all parameters are arrays
        the function returns a 2d array where 
        the entry at i,j correspoinds to the 
        coagulation coefficient for particle sizes
        dp1[i] and dp2[i] at temp[j] and pres[j].

        unit m3 s-1

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

def calc_coags(df,dp,temp,pres,dp_start=None):
    """ 
    Calculate coagulation sink

    Kulmala et al (2012): doi:10.1038/nprot.2012.091 

    Parameters
    ----------

    df : pandas.DataFrame
        Aerosol number size distribution
    dp : float or array
        Particle diameter(s) for which you want to calculate the CoagS, 
        unit: m
    temp : pandas.Series or float
        Ambient temperature corresponding to the data, unit: K
        If single value given it is used for all data
    pres : pandas.Series or float
        Ambient pressure corresponding to the data, unit: Pa
        If single value given it is used for all data
    dp_start : float, None 
        The smallest size that you consider as part of the coagulation sink
        If None (default) then the smallest size is from dp

    Returns
    -------
    
    pandas.DataFrame
        Coagulation sink for the given diamater(s),
        unit: s-1

    """

    if isinstance(temp,float):
        temp = pd.Series(index=df.index, data=temp)
    else:
        temp = temp.reindex(df.index, method="nearest")

    if isinstance(pres,float):
        pres = pd.Series(index=df.index, data=pres)
    else:
        pres = pres.reindex(df.index, method="nearest")

    if isinstance(dp,float):
        dp = [dp]

    temp = temp.values.reshape(-1,1)
    pres = pres.values.reshape(-1,1)
    
    coags = pd.DataFrame(index = df.index)
    i=0
    for dpi in dp:
        if dp_start is None:
            df = df.loc[:,df.columns.values.astype(float)>=dpi]
        elif dp_start<=dpi:
            df = df.loc[:,df.columns.values.astype(float)>=dpi]
        else:
            df = df.loc[:,df.columns.values.astype(float)>=dp_start]
        a = dndlogdp2dn(df)
        b = 1e6*coagulation_coef(dpi,df.columns.values.astype(float),temp,pres)
        coags.insert(i,dpi,(a*b).sum(axis=1,min_count=1))
        i+=1

    return coags
   
def diam2mob(dp,temp,pres,ne):
    """ 
    Convert electrical mobility diameter to electrical mobility in air

    Parameters
    ----------

    dp : float or numpy.array (m,)
        particle diameter(s),
        unit : m
    temp : float or numpy.array (n,1)
        ambient temperature, 
        unit: K
    pres : float or numpy.array (n,1)
        ambient pressure, 
        unit: Pa
    ne : int
        number of charges on the aerosol particle

    Returns
    -------

    float or numpy.array
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

    Zp : float
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

    float
        particle diameter, unit: m
    
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

    temp : float or numpy.array
        temperature, 
        unit: K
    pres : float or numpy.array
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

    float or numpy.array
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

    dp : float or numpy.array (m,)
        aerosol particle diameter(s), 
        unit: m
    temp : float or numpy.array (n,1)
        temperature, 
        unit: K
    pres : float or numpy.array (n,1)
        pressure,
        unit: Pa
    diffusivity : float or numpy.array (n,1)
        diffusivity of the gas that is condensing, 
        unit: m2/s
    molar_mass : float
        molar mass of the condensing gas, 
        unit: g/mol

    Returns
    -------

    float or numpy.array (n,m)
        Fuchs Sutugin correction factor for each particle diameter and 
        temperature/pressure 
        unit: m2/s

    """

    R = 8.314 
    l = 3.*diffusivity/((8.*R*temp)/(np.pi*molar_mass*0.001))**0.5
    knud = 2.*l/dp
    
    return (1. + knud)/(1. + 1.677*knud + 1.333*knud**2)

def calc_cs(df,temp,pres):
    """
    Calculate condensation sink, assuming that the condensing gas is sulfuric acid in air
    with aerosol particles.
    
    Kulmala et al (2012): doi:10.1038/nprot.2012.091 

    Parameters
    ----------

    df : pandas.DataFrame
        aerosol number size distribution (dN/dlogDp)
    temp : pandas.Series or float
        Ambient temperature corresponding to the data, unit: K
        If single value given it is used for all data
    pres : pandas.Series or float
        Ambient pressure corresponding to the data, unit: Pa
        If single value given it is used for all data

    Returns
    -------
    
    pandas.Series
        condensation sink time series, unit: s-1

    """
    
    if isinstance(temp,float):
        temp = pd.Series(index = df.index, data=temp)
    else:
        temp = temp.reindex(df.index, method="nearest")

    if isinstance(pres,float):
        pres = pd.Series(index = df.index, data=pres)
    else:
        pres = pres.reindex(df.index, method="nearest")

    temp = temp.values.reshape(-1,1)
    pres = pres.values.reshape(-1,1)

    M_h2so4 = 98.08   
    M_air = 28.965    
    V_air = 19.7      
    V_h2so4 = 51.96  

    dn = dndlogdp2dn(df)

    dp = df.columns.values.astype(float)

    diffu = binary_diffusivity(temp,pres,M_h2so4,M_air,V_h2so4,V_air)

    b = beta(dp,temp,pres,diffu,M_h2so4)

    df2 = (1e6*dn*(b*dp)).sum(axis=1,min_count=1)

    cs = (4.*np.pi*diffu)*df2.values.reshape(-1,1)

    return pd.Series(index=df.index, data=cs.flatten())

def calc_conc(df,dmin,dmax):
    """
    Calculate particle number concentration from aerosol 
    number-size distribution

    Parameters
    ----------

    df : pandas.DataFrame
        Aerosol number-size distribution
    dmin : float or array
        Size range lower diameter(s), unit: m
    dmax : float or array
        Size range upper diameter(s), unit: m

    Returns
    -------
    
    pandas.DataFrame
        Number concentration in the given size range(s), unit: cm-3

    """

    if isinstance(dmin,float):
        dmin = [dmin]
    if isinstance(dmax,float):
        dmax = [dmax]

    dp = df.columns.values.astype(float)
    conc_df = pd.DataFrame(index = df.index)

    for i in range(len(dmin)):
        dp1 = dmin[i]
        dp2 = dmax[i]
        findex = np.argwhere((dp<=dp2)&(dp>=dp1)).flatten()
        if len(findex)==0:
            conc = np.nan*np.ones(df.shape[0])
        else:
            dp_subset=dp[findex]
            conc=df.iloc[:,findex]
            logdp = calc_bin_edges(dp_subset)
            dlogdp = np.diff(logdp)
            conc = (conc*dlogdp).sum(axis=1, min_count=1)

        conc_df.insert(i,"N_%d" % i,conc)

    return conc_df

def calc_formation_rate(
    df,
    dp1,
    dp2,
    gr,
    temp,
    pres):
    """
    Calculate particle formation rate
    
    Kulmala et al (2012): doi:10.1038/nprot.2012.091

    Parameters
    ----------
    
    df : pd.DataFrame
        Aerosol particle number size distribution
    dp1 : float or array
        Lower diameter of the size range(s), unit: m
    dp2 : float or array
        Upper diameter of the size range(s), unit m
    gr : float or array
        Growth rate for particles out of the size range(s), 
        unit nm h-1
    temp : pandas.Series or float
        Ambient temperature corresponding to the data, unit: K
        If single value given it is used for all data
    pres : pandas.Series or float
        Ambient pressure corresponding to the data, unit: Pa
        If single value given it is used for all data

    Returns
    -------

    pandas.DataFrame
        particle formation rate(s) for the diameter range(s), unit: cm3 s-1

    """
    
    dn = dndlogdp2dn(df)

    dp = df.columns.values.astype(float)

    J = pd.DataFrame(index = df.index)

    for i in range(len(dp1)):
        idx = np.argwhere((dp>=dp1[i]) & (dp<=dp2[i])).flatten()

        # Sink term (consider all sizes inside the range) 
        sink_term = np.zeros(len(df.index))
        for j in idx:
            sink_term = sink_term + calc_coags(df,dp[j],temp,pres).values.flatten() * dn.iloc[:,j].values.flatten()
    
        # Conc term (observed change in the size range number concentration)
        dt = df.index.to_frame().diff().astype("timedelta64[s]").astype(float).values.flatten()
        dt[dt==0] = np.nan    
        conc = calc_conc(df,dp1[i],dp2[i])
        conc_term = conc.diff().values.flatten()/dt
    
        # GR term (consider the largest size in our size range)
        # GR is usually calculated for the size range 
        gr_term = (2.778e-13*gr[i])/(dp2[i]-dp1[i]) * dn.iloc[:,int(np.max(idx))].values.flatten()
        
        formation_rate = conc_term + sink_term + gr_term

        J.insert(i, "J_%d" % i, formation_rate)

    return J

def calc_ion_formation_rate(
    df_particles,
    df_negions,
    df_posions,
    dp1,
    dp2,
    gr_negions,
    gr_posions,
    temp,
    pres):
    """ 
    Calculate ion formation rate
    
    Kulmala et al (2012): doi:10.1038/nprot.2012.091

    Parameters
    ----------

    df_particles : pandas.DataFrame
         Aerosol particle number size distribution   
    df_negions : 
        Negative ion number size distribution
    df_posions : 
        Positive ion number size distribution
    dp1 : float or numpy.array
        Lower diameter of the size range(s), unit: m
    dp2 : float or numpy.array
        Upper diameter of the size range(s), unit: m
    gr_negions : float or numpy.array
        Growth rate for negative ions out of the size range(s), unit: nm h-1
    gr_posions : float or numpy.array
        Growth rate for positive ions out of the size range(s), unit: nm h-1
    temp : pandas.Series or float
        Ambient temperature corresponding to the data, unit: K
        If single value given it is used for all data
    pres : pandas.Series or float
        Ambient pressure corresponding to the data, unit: Pa
        If single value given it is used for all data

    Returns
    -------

    pandas.DataFrame
        Negative ion formation rate(s), unit : cm3 s-1
    pandas.DataFrame    
        Positive ion formation rate(s), unit: cm3 s-1

    """

    dn_particles = dndlogdp2dn(df_particles)
    dn_negions = dndlogdp2dn(df_negions)
    dn_posions = dndlogdp2dn(df_posions)

    dp = df_negions.columns.values.astype(float)
    time = df_negions.index

    J_negions = pd.DataFrame(index = df_negions.index)
    J_posions = pd.DataFrame(index = df_posions.index)

    # Constants
    alpha = 1.6e-6 # cm3 s-1
    Xi = 0.01e-6 # cm3 s-1

    for i in range(len(dp1)):
        idx = np.argwhere((dp>=dp1[i]) & (dp<=dp2[i])).flatten()

        # Sink terms
        sink_term_negions = np.zeros(len(time))
        sink_term_posions = np.zeros(len(time))
        for j in idx:
            sink_term_negions = sink_term_negions + calc_coags(df_particles,dp[j],temp,pres).values.flatten() * dn_negions.iloc[:,j].values.flatten()
            sink_term_posions = sink_term_posions + calc_coags(df_particles,dp[j],temp,pres).values.flatten() * dn_posions.iloc[:,j].values.flatten()

        # Conc terms
        dt = time.to_frame().diff().astype("timedelta64[s]").astype(float).values.flatten()
        dt[dt==0] = np.nan

        conc_negions = calc_conc(df_negions,dp1[i],dp2[i])
        conc_term_negions = conc_negions.diff().values.flatten()/dt

        conc_posions = calc_conc(df_posions,dp1[i],dp2[i])
        conc_term_posions = conc_posions.diff().values.flatten()/dt
 
        # GR terms
        gr_term_negions = (2.778e-13*gr_negions[i])/(dp2[i]-dp1[i]) * dn_negions.iloc[:,int(np.max(idx))].values.flatten()
        gr_term_posions = (2.778e-13*gr_posions[i])/(dp2[i]-dp1[i]) * dn_posions.iloc[:,int(np.max(idx))].values.flatten()

        # Recombination terms
        conc_small_negions = calc_conc(df_negions,0.5e-9,dp1[i])
        conc_small_posions = calc_conc(df_posions,0.5e-9,dp1[i])

        recombi_term_negions = alpha * conc_posions.values.flatten() * conc_small_negions.values.flatten()
        recombi_term_posions = alpha * conc_negions.values.flatten() * conc_small_posions.values.flatten()

        # Charging terms
        conc_particles = calc_conc(df_particles,dp1[i],dp2[i])
        charging_term_negions = Xi * conc_particles.values.flatten() * conc_small_negions.values.flatten()
        charging_term_posions = Xi * conc_particles.values.flatten() * conc_small_posions.values.flatten()

        formation_rate_negions = conc_term_negions + sink_term_negions + gr_term_negions + recombi_term_negions - charging_term_negions
        formation_rate_posions = conc_term_posions + sink_term_posions + gr_term_posions + recombi_term_posions - charging_term_posions

        J_negions.insert(i, "J_%d" % i, formation_rate_negions)
        J_posions.insert(i, "J_%d" % i, formation_rate_posions)

    return J_negions, J_posions

def tubeloss(diam, flowrate, tubelength, temp, pres):
    """
    Calculate diffusional particle losses to walls of
    straight cylindrical tube assuming a laminar flow regime

    Parameters
    ----------
    
    diam : numpy.array (m,)
        Particle diameters for which to calculate the
        losses, unit: m
    flowrate : numpy.array (n,)
        unit: L/min
    tubelength : float
        Length of the cylindrical tube
        unit: m
    temp : numpy.array (n,)
        temperature
        unit: K
    pres : numpy.array (n,)
        air pressure
        unit: Pa

    Returns
    -------

    numpy.array (n,m)
        Fraction of particles passing through.
        Each column represents diameter and each
        each row represents different temperature
        pressure and flowrate value
        
    """

    diameter_grid,temperature_grid = np.meshgrid(diam,temp)
    diameter_grid,pressure_grid = np.meshgrid(diam,pres)
    diameter_grid,sampleflow_grid = np.meshgrid(diam,flowrate)
    rmuu = np.pi*particle_diffusivity(diameter_grid,temperature_grid,pressure_grid)*tubelength/sampleflow_grid
    penetration = np.nan*np.ones(rmuu.shape)
    condition1 = (rmuu<0.02)
    condition2 = (rmuu>=0.02)
    penetration[condition1] = 1. - 2.56*rmuu[condition1]**(2./3.) + 1.2*rmuu[condition1]+0.177*rmuu[condition1]**(4./3.)
    penetration[condition2] = 0.819*np.exp(-3.657*rmuu[condition2]) + 0.097*np.exp(-22.3*rmuu[condition2]) + 0.032*np.exp(-57.0*rmuu[condition2])

    return penetration

def surf_dist(df):
    """
    Calculate the aerosol surface area size distribution

    Parameters
    ----------

    df : pandas.DataFrame
        Aerosol number-size distribution

    Returns
    -------
        
    pandas.DataFrame
        Aerosol surface area-size distribution
        unit: m2 cm-3

    """

    dp = df.columns.values.astype(float).flatten()

    return (np.pi*dp**2)*df

    
def vol_dist(df):
    """
    Calculate the aerosol volume size distribution

    Parameters
    ----------

    df : pandas.DataFrame
        Aerosol number-size distribution

    Returns
    -------
        
    pandas.DataFrame
        Aerosol volume-size distribution
        unit: m3 cm-3

    """
    dp = df.columns.values.astype(float).flatten()

    return (np.pi*(1./6.)*dp**3)*df

def calc_lung_df(dp):
    """
    Calculate lung deposition fractions for particle diameters

    ICRP, 1994. Human respiratory tract model for 
    radiological protection. A report of a task 
    group of the international commission on 
    radiological protection. Ann. ICRP 24 (1-3), 1-482

    Parameters
    ----------

    dp : array
        aerosol particle diameters
        unit: m

    Returns
    -------

    pandas.DataFrame
        Lung deposition fractions for alveoli ("DF_al"), trachea/bronchi ("DF_tb")
        head-airways ("DF_ha") and all combiend ("DF_tot")

    """

    # convert from meters to micrometers
    dp = dp*1e6

    # Deposition fractions
    IF = 1-0.5*(1.-1./(1.+0.00076*dp**2.8))
    DF_ha = IF*(1./(1.+np.exp(6.84+1.183*np.log(dp)))+1./(1.+np.exp(0.924-1.885*np.log(dp))))
    DF_al = (0.0155/dp)*(np.exp(-0.416*(np.log(dp)+2.84)**2) + 19.11*np.exp(-0.482*(np.log(dp)-1.362)**2))
    DF_tb = (0.00352/dp)*(np.exp(-0.234*(np.log(dp)+3.4)**2) + 63.9*np.exp(-0.819*(np.log(dp)-1.61)**2))
    DF_tot = IF*(0.0587 + 0.911/(1.+np.exp(4.77+1.485*np.log(dp)))+0.943/(1.+np.exp(0.508-2.58*np.log(dp)))) 

    DFs = pd.DataFrame({
        "DF_al":DF_al,
        "DF_tb":DF_tb,
        "DF_ha":DF_ha,
        "DF_tot":DF_tot
        })

    return DFs 

def calc_ldsa(df):
    """
    Calculate total LDSA from number size distribution data

    ICRP, 1994. Human respiratory tract model for 
    radiological protection. A report of a task 
    group of the international commission on 
    radiological protection. Ann. ICRP 24 (1-3), 1-482

    Parameters
    ----------
    
    df : pandas.DataFrame
        Aerosol number-size distribution

    Returns
    -------
    
    pandas.DataFrame
        Total LDSA for alveoli ("al"), trachea/bronchi ("tb")
        head-airways ("ha") and all combiend ("tot")
        unit: um2 cm-3
    
    """
    
    # m -> um
    dp = df.columns.values.astype(float)*1e6

    logdp = calc_bin_edges(dp)
    dlogdp = np.diff(logdp)

    # m2/cm-3 -> um2/cm-3
    surface_dist = surf_dist(df)*1e12

    # input needs ot be in m
    depo_fracs = calc_lung_df(dp*1e-6)

    ldsa_dist_al = surface_dist * depo_fracs.iloc[:,0].values.flatten()
    ldsa_dist_tb = surface_dist * depo_fracs.iloc[:,1].values.flatten()
    ldsa_dist_ha = surface_dist * depo_fracs.iloc[:,2].values.flatten()
    ldsa_dist_tot = surface_dist * depo_fracs.iloc[:,3].values.flatten()

    ldsa_dist = [ldsa_dist_al,ldsa_dist_tb,ldsa_dist_ha,ldsa_dist_tot]

    ldsa_column_names = ["LDSA_al","LDSA_tb","LDSA_ha","LDSA_tot"]

    df_ldsa = pd.DataFrame(index = df.index, columns = ldsa_column_names)

    for i in range(len(ldsa_dist)):
        ldsa = (ldsa_dist[i]*dlogdp).sum(axis=1,min_count=1)    
        df_ldsa[ldsa_column_names[i]] = ldsa

    return df_ldsa

def flow_velocity_in_pipe(tube_diam,flowrate):
    """
    Calculate fluid speed from the flow rate in circular tube
 
    Parameters
    ----------

    tube_diam : float or array
        Diameter of circular tube (m)
    flowrate : float or array
        Volumetric flow rate (lpm)

    Returns
    -------

    float or array
        Speed of fluid (m/s) 

    """
    
    volu_flow = flowrate/60000.
    cross_area = np.pi*(tube_diam/2.)**2
    
    return volu_flow/cross_area  

def pipe_reynolds(
    tube_diam,
    flowrate,
    temp,
    pres):
    """
    Calculate Reynolds number in a tube

    Parameters
    ----------

    tube_diam : float or array
        Inner diameter of the tube (m)
    flowrate : float or array
        Volumetric flow rate (lpm)
    temp : float or array
        Temperature in K
    pres : float or array
        Pressure in Pa

    Returns
    -------

    float or array
        Reynolds number

    """

    volu_flow = flowrate/60000.
    visc = air_viscosity(temp)
    dens = air_density(temp,pres)

    return (dens*volu_flow*tube_diam)/(visc*np.pi*(tube_diam/2.0)**2)

def dp2volts(thab_voltage,dp):
    """
    Convert particle diameters to DMA voltages

    Parameters
    ----------

    thab_voltage : float
        Voltage at THA+ peak (V)
    dp : float or array
        Particle diameters (m)

    Returns
    -------

    float or array:
        DMA voltage (V) corresponding to dp

    Notes
    -----
    
    Assumptions:

    1) Sheath flow is air
    2) Mobility standard used is THA+ monomer
    3) T = 293.15 K and p = 101325 Pa

    """

    thab_mob = (1.0/1.03)*1e-4
        
    Zp = diam2mob(dp,293.15,101325.0,1)
   
    return (thab_voltage * thab_mob)/Zp
