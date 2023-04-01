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
    
    logdp_mid = np.log10(df.columns.values.astype(float))
    logdp = (logdp_mid[:-1]+logdp_mid[1:])/2.0
    logdp = np.append(logdp,logdp_mid.max()+(logdp_mid.max()-logdp.max()))
    logdp = np.insert(logdp,0,logdp_mid.min()-(logdp.min()-logdp_mid.min()))
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

    """
   
    l = mean_free_path(temp,pres)
    return 1.+((2.*l)/dp)*(1.257+0.4*np.exp(-(1.1*dp)/(2.*l)))

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

def calc_coags(df,dp,temp,pres):
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
    
    coags = pd.DataFrame(index = df.index)
    i=0
    for dpi in dp:
        df = df.loc[:,df.columns.values.astype(float)>=dpi]
        a = dndlogdp2dn(df)
        b = 1e6*coagulation_coef(dpi,df.columns.values.astype(float),temp.values,pres.values)
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

    M_h2so4 = 98.08   
    M_air = 28.965    
    V_air = 19.7      
    V_h2so4 = 51.96  

    dn = dndlogdp2dn(df)

    dp = df.columns.values.astype(float)

    diffu = binary_diffusivity(temp.values,pres.values,M_h2so4,M_air,V_h2so4,V_air)

    b = beta(dp,temp.values,pres.values,diffu,M_h2so4)

    df2 = (1e6*dn*(b*dp)).sum(axis=1,min_count=1)

    cs = (4.*np.pi*diffu)*df2.values

    return pd.Series(index=df.index,data=cs)

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
            logdp_mid=np.log10(dp_subset)
            logdp=(logdp_mid[:-1]+logdp_mid[1:])/2.0
            logdp=np.append(logdp,logdp_mid.max()+(logdp_mid.max()-logdp.max()))
            logdp=np.insert(logdp,0,logdp_mid.min()-(logdp.min()-logdp_mid.min()))
            dlogdp=np.diff(logdp)
            conc=np.nansum(conc*dlogdp,axis=1)

        conc_df.insert(i,"%.2e_%.2e" % (dp1,dp2),conc)

    return conc_df

def calc_formation_rate(
    dp1,
    dp2,
    conc,
    coags,
    gr):
    """
    Calculate particle formation rate
    
    Kulmala et al (2012): doi:10.1038/nprot.2012.091

    Parameters
    ----------
    
    dp1 : float or array
        Lower diameter of the size range(s), unit: m
    dp2 : float or array
        Upper diameter of the size range(s), unit m
    conc : pandas.DataFrame
        particle number concentration timeseries
        in the size range(s), unit cm-3
    coags : pandas.DataFrame
        Coagulation sink timeseries for particles 
        in the size range(s). unit s-1 

        Usually approximated as coagulation sink for particle size
        in the lower limit of the size range, 
        unit s-1
    gr : float or array
        Growth rate for particles out of the size range(s), 
        unit nm h-1

    Returns
    -------

    pandas.DataFrame
        particle formation rate for diameter(s), unit: cm3 s-1

    """

    # Fit the coags to the conc index
    coags = coags.reindex(conc.index,method="nearest")

    # Construct the dt frame
    dt = conc.index.to_frame().diff().astype("timedelta64[s]").astype(float)

    conc_term = conc.diff().values/dt.values
    sink_term = coags.values * conc.values
    gr_term = (2.778e-13*gr)/(dp2-dp1) * conc.values
    formation_rate = conc_term + sink_term + gr_term
    
    J = pd.DataFrame(data = formation_rate, index = conc.index, columns=coags.columns)

    return J

def calc_ion_formation_rate(
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

    dp1 : float or numpy.array
        Lower diameter of the size range(s), unit: m
    dp2 : float or numpy.array
        Upper diameter of the size range(s), unit: m
    conc_pos : pandas.DataFrame
        Positive ion number concentration in the size range(s), unit: cm-3. 
        Each size range corresponds to a column in the dataframe
    conc_neg : pandas.DataFrame
        Negative ion number concentration in the size range(s), unit: cm-3
    conc_pos_small : pandas.DataFrame
        Positive ion number concentration for ions smaller than size range(s), unit: cm-3
    conc_neg_small : pandas.DataFrame
        Negative ion number concentration for ions smaller than size range(s), unit: cm-3
    conc : pandas.DataFrame
        Particle number concentration in the size range(s), unit: cm-3
    coags : pandas.DataFrame
        Coagulation sink for particles in the size range(s).
        unit: s-1
    gr : float or numpy.array
        Growth rate for particles out of the size range(s), unit: nm h-1

    Returns
    -------

    pandas.DataFrame
        Negative ion formation rate(s), unit : cm3 s-1
    pandas.DataFrame    
        Positive ion formation rate(s), unit: cm3 s-1

    """

    # Reindex everything to conc_neg
    coags = coags.reindex(conc_neg.index,method="nearest")
    conc_pos = conc_pos.reindex(conc_neg.index,method="nearest")
    conc = conc.reindex(conc_neg.index,method="nearest")
    conc_neg_small = conc_neg_small.reindex(conc_neg.index,method="nearest")
    conc_pos_small = conc_pos_small.reindex(conc_neg.index,method="nearest")

    # Constants
    alpha = 1.6e-6 # cm3 s-1
    Xi = 0.01e-6 # cm3 s-1

    # Construct the dt frame
    dt = conc_neg.index.to_frame().diff().astype("timedelta64[s]").astype(float)

    # Calculate the terms
    pos_conc_term = conc_pos.diff().values/dt.values
    pos_sink_term = coags.values * conc_pos.values
    pos_gr_term = (2.778e-13*gr)/(dp2-dp1) * conc_pos.values
    pos_recombination_term = alpha * conc_pos.values * conc_neg_small.values
    pos_charging_term = Xi * conc.values * conc_pos_small.values
    pos_formation_rate = pos_conc_term + pos_sink_term + pos_gr_term + pos_recombination_term - pos_charging_term

    J_pos = pd.DataFrame(data=pos_formation_rate,columns=coags.columns,index=conc_neg.index)

    neg_conc_term = conc_neg.diff().values/dt.values
    neg_sink_term = coags.values * conc_neg.values
    neg_gr_term = (2.778e-13*gr)/(dp2-dp1) * conc_neg.values
    neg_recombination_term = alpha * conc_neg.values * conc_pos_small.values
    neg_charging_term = Xi * conc.values * conc_neg_small.values
    neg_formation_rate = neg_conc_term + neg_sink_term + neg_gr_term + neg_recombination_term - neg_charging_term

    J_neg = pd.DataFrame(data=neg_formation_rate,columns=coags.columns,index=conc_neg.index)

    return J_neg, J_pos

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
