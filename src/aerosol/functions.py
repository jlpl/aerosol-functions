import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dts
from matplotlib import colors
from matplotlib.pyplot import cm
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from astral import Observer
from astral.sun import noon
from scipy.signal import correlate, correlation_lags, fftconvolve


# All constants are SI base units
E=1.602E-19           # elementary charge
E_0=8.85418781e-12    # permittivity of vacuum
K_B=1.381e-23         # Boltzmann constant 
R=8.3413              # gas constant

# Helper functions
def is_input_float(args):
    for arg in args:
        if (isinstance(arg,float) | isinstance(arg,int)) is False:
            return False
    return True

def get_index(args):
    longest_series = max(args,key=len)
    return longest_series.index

def check_lengths(series_list):
    lengths = [len(series) for series in series_list if len(series) > 1]
    return all(length == lengths[0] for length in lengths)

def air_density(temp,pres):
    """
    Calculate air density

    Parameters
    ----------

    temp : float or series of lenght n
        absolute temperature (K) 
    pres : float or series of length n
        absolute pressure (Pa)
 
    Returns
    -------

    float or series of length n
        air density (kg/m3)
        
    """

    float_input = is_input_float([temp,pres])

    pres = pd.Series(pres)
    temp = pd.Series(temp)

    assert check_lengths([pres,temp]), "length mismatch"

    idx = get_index([temp,pres])

    dens = pres.values/(287.0500676*temp.values)

    if float_input:
        return dens[0]
    else:
        return pd.Series(index = idx, data = dens)

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

    return pd.to_datetime(datetime.fromordinal(int(datenum)) + 
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
    Calculate bin edges given bin centers
    
    Parameters
    ----------
    
    dp : pandas series of lenght n
        bin center diameters

    Returns
    -------

    pandas series of lenght n+1
        log bin edges

    """
    dp_arr = dp.values
    logdp_mid = np.log10(dp_arr)
    logdp = (logdp_mid[:-1]+logdp_mid[1:])/2.0
    maxval = [logdp_mid.max()+(logdp_mid.max()-logdp.max())]
    minval = [logdp_mid.min()-(logdp.min()-logdp_mid.min())]
    logdp = np.concatenate((minval,logdp,maxval))
    
    return pd.Series(logdp)

def dndlogdp2dn(df):
    """    
    Convert from normalized number concentrations to
    unnormalized number concentrations.

    Parameters
    ----------

    df : dataframe
        Aerosol number-size distribution (dN/dlogDp)

    Returns
    -------

    dataframe
        Aerosol number size distribution (dN)

    """
    
    dp = df.columns.values.astype(float)
    logdp = calc_bin_edges(pd.Series(dp))
    dlogdp = np.diff(logdp) #this will be 1d numpy array

    return df*dlogdp

def air_viscosity(temp):
    """ 
    Calculate air viscosity using Enskog-Chapman theory

    Parameters
    ----------

    temp : float or series of length n
        air temperature, unit: K  

    Returns
    -------

    float or series of length n
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

    temp : float or series of length n
        air temperature, unit: K  
    pres : float or series of length n
        air pressure, unit: Pa

    Returns
    -------

    float or series of length n
        mean free path in air, unit: m

    """

    float_input = is_input_float([temp,pres])

    pres = pd.Series(pres)
    temp = pd.Series(temp)

    assert check_lengths([pres,temp]), "length mismatch"

    idx = get_index([temp,pres])

    Mair=0.02897
    mu=air_viscosity(temp)

    l = (mu.values/pres.values)*((np.pi*R*temp.values)/(2.*Mair))**0.5

    if float_input:
        return l[0]
    else:
        return pd.Series(index=idx,data=l)

def slipcorr(dp,temp=293.15,pres=101325.):
    """
    Slip correction factor in air 

    Parameters
    ----------

    dp : float or series of lenght m
        particle diameter, unit m 
    temp : float or series of length n
        air temperature, unit K 
    pres : float or series of lenght n
        air pressure, unit Pa

    Returns
    -------

    float or dataframe fo shape (n,m)
        For dataframe the index is taken from temperature
        or pressure series. Columns are particle diameters.
        unit dimensionless


    Notes
    -----

    Correction is done according to Mäkelä et al. (1996)

    """
   
    float_input = is_input_float([dp,temp,pres])

    dp = pd.Series(dp)
    temp = pd.Series(temp)
    pres = pd.Series(pres)

    assert check_lengths([pres,temp]), "length mismatch"

    idx = get_index([temp,pres])

    l = mean_free_path(temp,pres).values.reshape(-1,1)
    dp = dp.values
    cc = 1.+((2.*l)/dp)*(1.246+0.420*np.exp(-(0.87*dp)/(2.*l)))

    if float_input:
        return cc[0][0]
    else:
        return pd.DataFrame(index = idx, columns = dp, data = cc)

def particle_diffusivity(dp,temp=293.15,pres=101325.):
    """ 
    Particle brownian diffusivity in air 

    Parameters
    ----------

    dp : float or series of lenght m
        particle diameter, unit: m 
    temp : float or series of lenght n
        air temperature, unit: K 
    pres : float or series of lenght n
        air pressure, unit: Pa

    Returns
    -------

    float or dataframe of shape (n,m)
        Particle Brownian diffusivity in air
        unit m2 s-1

    """

    float_input = is_input_float([dp,temp,pres])

    dp = pd.Series(dp)
    temp = pd.Series(temp)
    pres = pd.Series(pres)

    idx = get_index([temp,pres])

    cc = slipcorr(dp,temp,pres)
    mu = air_viscosity(temp)

    cc = cc.values
    dp = dp.values
    temp = temp.values.reshape(-1,1)
    mu = mu.values.reshape(-1,1)

    D = (K_B*temp*cc)/(3.*np.pi*mu*dp) 

    if float_input:
        return D[0][0]
    else:
        return pd.DataFrame(index = idx, columns = dp, data = D)

def particle_thermal_speed(dp,temp):
    """
    Particle thermal speed 

    Parameters
    ----------

    dp : float or series
        particle diameter, unit: m 
    temp : float or series
        air temperature, unit: K 

    Returns
    -------

    float or dataframe
        Particle thermal speed 
        point, unit: m s-1

    """

    float_input = is_input_float([dp,temp])

    dp = pd.Series(dp)
    temp = pd.Series(temp)

    idx = temp.index

    rho_p = 1000.0
    mp = rho_p*(1./6.)*np.pi*dp**3.

    dp = dp.values
    mp = mp.values
    temp = temp.values.reshape(-1,1)

    vp=((8.*K_B*temp)/(np.pi*mp))**(1./2.)

    if float_input:
        return vp[0][0]
    else:
        return pd.DataFrame(index = idx, columns = dp, data = vp)

def particle_mean_free_path(dp,temp=293.15,pres=101325.):
    """ 
    Particle mean free path in air 

    Parameters
    ----------

    dp : float or series of length m
        particle diameter, unit: m 
    temp : float or series of length n
        air temperature, unit: K 
    pres : float or series of length n
        air pressure, unit: Pa

    Returns
    -------

    float or dataframe of shape (n,m)
        Particle mean free path, unit: m

    """

    float_input = is_input_float([dp,temp,pres])

    dp = pd.Series(dp)
    temp = pd.Series(temp)
    pres = pd.Series(pres)

    idx = get_index([temp,pres])

    D=particle_diffusivity(dp,temp,pres)
    c=particle_thermal_speed(dp,temp)

    v_therm = (8.*D.values)/(np.pi*c.values)

    if float_input:
        return v_therm[0][0]
    else:
        return pd.DataFrame(index=idx, columns=dp, data=v_therm)

def coagulation_coef(dp1,dp2,temp=293.15,pres=101325.):
    """ 
    Calculate Brownian coagulation coefficient (Fuchs)

    Parameters
    ----------

    dp1 : float
        first particle diameter, unit: m 
    dp2 : float or series of lenght m
        second particle diameter, unit: m 
    temp : float or series of lenght n
        air temperature, unit: K 
    pres : float or series of lenght n
        air pressure, unit: Pa

    Returns
    -------

    float or dataframe of shape (n,m)
        Brownian coagulation coefficient (Fuchs),

        If dataframe is returned the columns correspond
        to diameter pairs (dp1,dp2) and are labeled by 
        elements in dp2.

        unit m3 s-1

    """

    # Is it all float input?
    float_input = is_input_float([dp2,temp,pres])

    # Convert everything to series for the calculations
    dp1 = pd.Series(dp1)
    dp2 = pd.Series(dp2)
    temp = pd.Series(temp)
    pres = pd.Series(pres)

    idx = get_index([temp,pres])

    def particle_g(dp,temp,pres):
        l = particle_mean_free_path(dp,temp,pres).values
        dp = dp.values
        return 1./(3.*dp*l)*((dp+l)**3.-(dp**2.+l**2.)**(3./2.))-dp

    D1 = particle_diffusivity(dp1,temp,pres).values
    D2 = particle_diffusivity(dp2,temp,pres).values
    g1 = particle_g(dp1,temp,pres)
    g2 = particle_g(dp2,temp,pres)
    c1 = particle_thermal_speed(dp1,temp).values
    c2 = particle_thermal_speed(dp2,temp).values

    dp1 = dp1.values
    dp2 = dp2.values

    coag_coef = 2.*np.pi*(D1+D2)*(dp1+dp2) \
           * 1./( (dp1+dp2)/(dp1+dp2+2.*(g1**2.+g2**2.)**0.5) + \
           +   (8.*(D1+D2))/((c1**2.+c2**2.)**0.5*(dp1+dp2)) )

    if float_input:
        return coag_coef[0][0]
    else:
        return pd.DataFrame(index = idx, columns=dp2, data=coag_coef)


def calc_coags(df,dp,temp=293.15,pres=101325.,dp_start=None):
    """ 
    Calculate coagulation sink

    Kulmala et al (2012): doi:10.1038/nprot.2012.091 

    Parameters
    ----------

    df : dataframe
        Aerosol number size distribution
    dp : float or series of length m
        Particle diameter(s) for which you want to calculate the CoagS, 
        unit: m
    temp : float or series indexed by DatetimeIndex
        Ambient temperature corresponding to the data, unit: K
        If single value given it is used for all data
    pres : float or series indexed by DatetimeIndex
        Ambient pressure corresponding to the data, unit: Pa
        If single value given it is used for all data
    dp_start : float or None
        The smallest size that you consider as part of the coagulation sink
        If None (default) then the smallest size is from dp

    Returns
    -------
    
    float or dataframe
        Coagulation sink for the given diamater(s),
        unit: s-1

    """

    # index is now taken from the size distribution

    temp=pd.Series(temp)
    pres=pd.Series(pres)

    if len(temp)==1:
        temp = pd.Series(index=df.index, data=temp.values[0])
    else:
        temp = temp.reindex(df.index, method="nearest")

    if len(pres)==1:
        pres = pd.Series(index=df.index, data=pres.values[0])
    else:
        pres = pres.reindex(df.index, method="nearest")

    dp = pd.Series(dp)

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
        b = 1e6*coagulation_coef(dpi,pd.Series(df.columns.values.astype(float)),temp,pres)
        c = pd.DataFrame(a.values*b.values).sum(axis=1,min_count=1)
        coags.insert(i,dpi,c.values)
        i+=1

    return coags

def cs2coags(cs,dp,m=-1.6):
    """
    Estimate coagulation sink from condensation sink

    Parameters
    ----------

    cs : pandas.Series
        The condensation sink time series: unit s-1
    dp : float
        Particle diameter for which CoagS is calculated, unit: nm
    m : float
        Exponent in the equation

    Returns
    -------

    coags : pandas.Series
        Coagulation sink time series for size dp

    References
    ----------

    Kulmala et al (2012), doi:10.1038/nprot.2012.091

    """

    return cs * (dp/0.71)**m



def diam2mob(dp,temp=293.15,pres=101325.0,ne=1):
    """ 
    Convert electrical mobility diameter to electrical mobility in air

    Parameters
    ----------

    dp : float
        particle diameter(s),
        unit : nm
    temp : float
        ambient temperature
        default 20 C 
        unit: K
    pres : float
        ambient pressure,
        default 1 atm 
        unit: Pa
    ne : int
        number and polarity of charges on the aerosol particle
        default 1

    Returns
    -------

    float
        particle electrical mobility, 
        unit: cm2 s-1 V-1

    """

    cc = slipcorr(dp*1e-9,temp,pres) # dataframe
    mu = air_viscosity(temp) # series

    Zp = (ne*E*cc)/(3.*np.pi*mu*dp*1e-9)*1e4

    return Zp

def mob2diam(Zp,temp=293.15,pres=101325.,ne=1, tol=1e-3, maxiter=100):
    """
    Convert electrical mobility to electrical mobility diameter in air

    Parameters
    ----------

    Zp : float
        particle electrical mobility or mobilities, 
        unit: cm2 s-1 V-1
    temp : float
        ambient temperature, 
        unit: K
    pres : float
        ambient pressure, 
        unit: Pa
    ne : integer
        number and polarity of elementary charges on the aerosol particle

    Returns
    -------

    float
        particle diameter, unit: m
    
    """
    
    ne = np.abs(ne)
    Zp = np.abs(Zp)

    def minimize_this(dp,Z):
        return np.abs(diam2mob(dp,temp,pres,ne)-Z)

    # Initial guessing
    if (Zp>0.1):
        dp0=1.0
    elif (Zp>0.001):
        dp0=10.0
    elif (Zp>=0.001):
        dp0=50.0
    elif (Zp>=0.0001):
        dp0=150.0
    elif (Zp>=0.00001):
        dp0=1000.0
    else:
        dp0=10000.0

    # Optimization using Nelder Mead method
    diam = minimize(minimize_this, 
        dp0, 
        args=(Zp,), 
        tol=tol, 
        method='Nelder-Mead',
        options={"maxiter":maxiter})

    if not diam.success:
        return np.nan
    else:
        return diam.x[0]


def binary_diffusivity(temp,pres,Ma,Mb,Va,Vb):
    """ 
    Binary diffusivity in a mixture of gases a and b

    Fuller et al. (1966): https://doi.org/10.1021/ie50677a007 

    Parameters
    ----------

    temp : float or series of length n
        temperature, 
        unit: K
    pres : float or series of length n
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

    float or series of length n
        binary diffusivity, 
        unit: m2 s-1

    """

    float_input = is_input_float([temp,pres])

    temp = pd.Series(temp)
    pres = pd.Series(pres)

    idx = get_index([temp,pres])
    
    diffusivity = (1.013e-2*(temp.values**1.75)*np.sqrt((1./Ma)+(1./Mb)))/(pres.values*(Va**(1./3.)+Vb**(1./3.))**2)
    
    if float_input:
        return diffusivity[0]
    else:
        return pd.Series(index = idx, data = diffusivity)


def beta(dp,temp,pres,diffusivity,molar_mass):
    """ 
    Calculate Fuchs Sutugin correction factor 

    Sutugin et al. (1971): https://doi.org/10.1016/0021-8502(71)90061-9

    Parameters
    ----------

    dp : float or series of lenght m
        aerosol particle diameter(s), 
        unit: m
    temp : float or series of lenght n
        temperature, 
        unit: K
    pres : float or series of lenght n
        pressure,
        unit: Pa
    diffusivity : float or series of length n
        diffusivity of the gas that is condensing, 
        unit: m2/s
    molar_mass : float
        molar mass of the condensing gas, 
        unit: g/mol

    Returns
    -------

    float or dataframe of shape (n,m)
        Fuchs Sutugin correction factor for each particle diameter and 
        temperature/pressure 
        unit: m2/s

    """

    float_input = is_input_float([dp,temp,pres,diffusivity])

    dp = pd.Series(dp)
    temp = pd.Series(temp)
    pres = pd.Series(pres)
    diffusivity = pd.Series(diffusivity)

    idx = get_index([temp,pres])

    dp = dp.values
    temp = temp.values.reshape(-1,1)
    pres = pres.values.reshape(-1,1)
    diffusivity = diffusivity.values.reshape(-1,1)

    l = 3.*diffusivity/((8.*R*temp)/(np.pi*molar_mass*0.001))**0.5
    
    knud = 2.*l/dp
    
    b = (1. + knud)/(1. + 1.677*knud + 1.333*knud**2)

    if float_input:
        return b[0][0]
    else:
        return pd.DataFrame(index=idx,columns=dp,data=b)


def calc_cs(df,temp=293.15,pres=101325.):
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
        condensation sink, unit: s-1

    """

    temp = pd.Series(temp)
    pres = pd.Series(pres)

    if len(temp)==1:
        temp = pd.Series(index = df.index, data = temp.values[0])
    else:
        temp = temp.reindex(df.index, method="nearest")

    if len(pres)==1:
        pres = pd.Series(index = df.index, data = pres.values[0])
    else:
        pres = pres.reindex(df.index, method="nearest")

    M_h2so4 = 98.08   
    M_air = 28.965    
    V_air = 19.7      
    V_h2so4 = 51.96  

    dn = dndlogdp2dn(df) # dataframe

    dp = pd.Series(df.columns.values.astype(float)) #series

    diffu = binary_diffusivity(temp,pres,M_h2so4,M_air,V_h2so4,V_air) #series

    b = beta(dp,temp,pres,diffu,M_h2so4) #dataframe

    df2 = pd.DataFrame(1e6*dn.values*(b.values*dp.values)).sum(axis=1,min_count=1) #dataframe

    cs = (4.*np.pi*diffu.values)*df2.values

    return pd.Series(index = df.index, data = cs)


def calc_conc(df,dmin,dmax,frac=0.5):
    """
    Calculate particle number concentration from aerosol
    number-size distribution by adding whole bins

    Parameters
    ----------

    df : dataframe
        Aerosol number-size distribution
    dmin : float or series of length n
        Size range lower diameter(s), unit: m
    dmax : float or series of length n
        Size range upper diameter(s), unit: m
    frac : float
        Minimum fraction of available data when calculating a concentration point

    Returns
    -------

    dataframe
        Number concentration in the given size range(s), unit: cm-3

    """

    dmin = pd.Series(dmin)
    dmax = pd.Series(dmax)

    dp = df.columns.values.astype(float)
    conc_df = pd.DataFrame(index = df.index)

    for i in range(len(dmin)):
        dp1 = dmin.values[i]
        dp2 = dmax.values[i]
        findex = np.argwhere((dp<=dp2)&(dp>=dp1)).flatten()
        if len(findex)==0:
            conc = np.nan*np.ones(df.shape[0])
        else:
            dp_subset=dp[findex]
            conc=df.iloc[:,findex]
            logdp = calc_bin_edges(pd.Series(dp_subset))
            dlogdp = np.diff(logdp)
            conc = (conc*dlogdp).sum(axis=1, min_count=int(frac*len(findex)))

        conc_df.insert(i,i,conc)

    return conc_df

def filter_nans(df, threshold=0.0, axis=1):
    if (axis==0):
        df_filt = df.iloc[:,(df.isnull().mean(axis=axis)<=threshold).values]
    if (axis==1):
        df_filt = df.iloc[(df.isnull().mean(axis=axis)<=threshold).values,:]

    if isinstance(df_filt, pd.Series):
        return df_filt.to_frame()
    else:
        return df_filt

def calc_conc_interp(df,dmin,dmax,threshold=0.0):
    """
    Calculate particle number concentration from aerosol
    number-size distribution using integration and linear 
    approximation

    Parameters
    ----------

    df : dataframe
        Aerosol number-size distribution
    dmin : float
        Size range lower diameter, unit: m
    dmax : float
        Size range upper diameter, unit: m
    threshold : float
        fraction of nans accepted per row

    Returns
    -------

    series
        Number concentration in the given size range, unit: cm-3
        in the index of the original dataframe 

    Note
    ----

    If you provide volume or surface size distribution the result
    will be volume concentration or surface concentration

    """

    #dmin = pd.Series(dmin)
    #dmax = pd.Series(dmax)

    # Find the columns with dmin and dmax
    dp = df.columns.astype(float).values

    findex_min = int(np.max([0,np.min(np.argwhere(dp>dmin))-1]))
    findex_max = int(np.min([len(dp)-1,np.max(np.argwhere(dp<dmax))+1]))

    df = df.iloc[:,findex_min:findex_max+1]

    # Only keep rows with more than threshold fraction of nan-values
    df_filt = filter_nans(df,threshold=threshold,axis=1)

    if df_filt.empty:
        return pd.Series(index = df.index)

    # Interpolate away the nans
    df_filt = df_filt.interpolate(limit_area="inside",axis=1).dropna(how="all",axis=1).interpolate(axis=1)

    logdp = np.log10(df_filt.columns.astype(float).values)
    data = df_filt.values

    min_logdp = np.min(logdp)
    max_logdp = np.max(logdp)

    dp_grid = np.linspace(min_logdp,max_logdp,1000)

    data_interp = np.nan*np.ones((data.shape[0],len(dp_grid)))

    for j in range(data.shape[0]):
        data_interp[j,:] = np.interp(dp_grid,logdp,data[j,:])

    conc = trapezoid(data_interp, x = dp_grid, axis=1)

    conc_s = pd.Series(index = df_filt.index, data = conc)

    return conc_s.reindex(df.index)

def calc_formation_rate(
    df,
    dp1,
    dp2,
    gr,
    sink_term):
    """
    Calculate particle formation rate
    
    Kulmala et al (2012): doi:10.1038/nprot.2012.091

    Parameters
    ----------
    
    df : Dataframe (m rows)
        Aerosol particle number size distribution 
        Unit cm-3
    dp1 : float or series of length n
        Lower diameter of the size range(s)
        Unit m
    dp2 : float or series of length n
        Upper diameter of the size range(s)
        Unit m
    gr : float or Dataframe (n columns, m rows)
        Growth rates
        unit nm/h
    sink_term : Dataframe (n columns, m rows) 
        Flux of particles out of size range due to sink
        Unit: cm-3 s-1

    Returns
    -------

    dict
        Particle formation rate, dN/dt terms, sink terms
        and GR terms for the diameter range(s)
        Unit cm-3 s-1

    """
    
    j_terms = pd.DataFrame(index = df.index)
    gr_terms = pd.DataFrame(index = df.index)
    sink_terms = pd.DataFrame(index = df.index)
    conc_terms = pd.DataFrame(index = df.index)

    dp1 = pd.Series(dp1).values
    dp2 = pd.Series(dp2).values

    if is_input_float([gr]):
        gr = pd.DataFrame([gr])

    for i in range(len(dp1)):
        conc = calc_conc(df,dp1[i],dp2[i])

        # Conc term
        dt = df.index.to_frame().diff().values.astype("timedelta64[s]").astype(float).flatten()
        dt[dt<=0] = np.nan    
        
        conc_term = conc.diff().values.flatten()/dt 

        # Sink term
        sink_term = sink_term.iloc[:,i].values.flatten()

        # GR term
        gr_term = (2.778e-13*gr.iloc[:,i].values.flatten())/(dp2[i]-dp1[i]) * conc.values.flatten()
        
        # Formation rate
        formation_rate = conc_term + sink_term + gr_term

        j_terms.insert(i,i,formation_rate)
        gr_terms.insert(i,i,gr_term)
        sink_terms.insert(i,i,sink_term)
        conc_terms.insert(i,i,conc_term)

    return {"J":j_terms,"conc":conc_terms,"sink":sink_terms,"GR":gr_terms}

def calc_ion_formation_rate(
    df_particles,
    df_negions,
    df_posions,
    dp1,
    dp2,
    gr_negions,
    gr_posions,
    sink_term_negions,
    sink_term_posions):
    """ 
    Calculate ion formation rate
    
    Kulmala et al (2012): doi:10.1038/nprot.2012.091

    Parameters
    ----------

    df_particles : dataframe with m rows
         Aerosol particle number size distribution   
    df_negions : dataframe with m rows
        Negative ion number size distribution
    df_posions : dataframe with m rows
        Positive ion number size distribution
    dp1 : float or series of length n
        Lower diameter of the size range(s), unit: m
    dp2 : float or series of length n
        Upper diameter of the size range(s), unit: m
    gr_negions : Float or DataFrame (n columns and m rows)
        The negative ion GRs
        unit nm h-1
    gr_posions : Float or DataFrame (n columns and m rows)
        The positive ion GRs
        unit nm h-1
    sink_term_negions : Dataframe (n columns and m rows)
        Flux of negative ions out of the size range due to coagulation
        Unit: cm-3 s-1
    sink_term_posions : Dataframe (n columns and m rows)
        Flux of positive ions out of the size range due to coagulation
        Unit: cm-3 s-1
 
    Returns
    -------

    dict
        Negative ion formation rate, dN/dt terms, sink terms,
        GR terms, attachment terms and recombination terms 
        for the diameter range(s)
        Unit cm-3 s-1
    dict  
        Positive ion formation rate, dN/dt terms, sink terms,
        GR terms, attachment terms and recombination terms 
        for the diameter range(s)
        Unit cm-3 s-1
 
    """
    
    time = df_negions.index

    j_terms_negions = pd.DataFrame(index = df_negions.index)
    j_terms_posions = pd.DataFrame(index = df_posions.index)
    conc_terms_negions = pd.DataFrame(index = df_negions.index)
    conc_terms_posions = pd.DataFrame(index = df_posions.index)
    sink_terms_negions = pd.DataFrame(index = df_negions.index)
    sink_terms_posions = pd.DataFrame(index = df_posions.index)
    gr_terms_negions = pd.DataFrame(index = df_negions.index)
    gr_terms_posions = pd.DataFrame(index = df_posions.index)
    charging_terms_negions = pd.DataFrame(index = df_negions.index)
    charging_terms_posions = pd.DataFrame(index = df_posions.index)
    recombi_terms_negions = pd.DataFrame(index = df_negions.index)
    recombi_terms_posions = pd.DataFrame(index = df_posions.index)

    # Constants
    alpha = 1.6e-6 # cm3 s-1
    Xi = 0.01e-6 # cm3 s-1

    dp1=pd.Series(dp1).values
    dp2=pd.Series(dp2).values

    if is_input_float([gr_negions,gr_posions]):
        gr_negions = pd.DataFrame([gr_negions])
        gr_posions = pd.DataFrame([gr_posions])

    for i in range(len(dp1)):
        conc_negions = calc_conc(df_negions,dp1[i],dp2[i])
        conc_posions = calc_conc(df_posions,dp1[i],dp2[i])

        # Sink terms
        sink_term_negions = sink_term_negions.iloc[:,i].values.flatten()
        sink_term_posions = sink_term_posions.iloc[:,i].values.flatten()

        # Conc terms
        dt = time.to_frame().diff().values.astype("timedelta64[s]").astype(float).flatten()
        dt[dt<=0] = np.nan
        conc_term_negions = conc_negions.diff().values.flatten()/dt
        conc_term_posions = conc_posions.diff().values.flatten()/dt
 
        # GR terms
        gr_term_negions = (2.778e-13*gr_negions.iloc[:,i].values.flatten())/(dp2[i]-dp1[i]) * conc_negions.values.flatten()
        gr_term_posions = (2.778e-13*gr_posions.iloc[:,i].values.flatten())/(dp2[i]-dp1[i]) * conc_posions.values.flatten()

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

        j_terms_negions.insert(i, i, formation_rate_negions)
        j_terms_posions.insert(i, i, formation_rate_posions)
        conc_terms_negions.insert(i, i, conc_term_negions)
        conc_terms_posions.insert(i, i, conc_term_posions)
        sink_terms_negions.insert(i, i, sink_term_negions)
        sink_terms_posions.insert(i, i, sink_term_posions)
        gr_terms_negions.insert(i, i, gr_term_negions)
        gr_terms_posions.insert(i, i, gr_term_posions)
        charging_terms_negions.insert(i, i, charging_term_negions)
        charging_terms_posions.insert(i, i, charging_term_posions)
        recombi_terms_negions.insert(i, i, recombi_term_negions)
        recombi_terms_posions.insert(i, i, recombi_term_posions)

    results_negions = {
        "J":j_terms_negions,
        "conc":conc_terms_negions,
        "sink":sink_terms_negions,
        "GR":gr_terms_negions,
        "Attach":charging_terms_negions,
        "Recombi":recombi_terms_negions
        }

    results_posions = {
        "J":j_terms_posions,
        "conc":conc_terms_posions,
        "sink":sink_terms_posions,
        "GR":gr_terms_posions,
        "Attach":charging_terms_posions,
        "Recombi":recombi_terms_posions 
        }

    return results_negions, results_posions

def tubeloss(diam, flowrate, tubelength, temp=293.15, pres=101325.):
    """
    Calculate diffusional particle losses to walls of
    straight cylindrical tube assuming a laminar flow regime

    Parameters
    ----------
    
    diam : float or series of length m
        Particle diameters for which to calculate the
        losses, unit: m
    flowrate : float or series of length n
        unit: L/min
    tubelength : float
        Length of the cylindrical tube
        unit: m
    temp : float or series of length n
        temperature
        unit: K
    pres : float or series of lenght n
        air pressure
        unit: Pa

    Returns
    -------

    float or dataframe of shape (n,m)
        Fraction of particles passing through.
        Each column represents diameter and each
        each row represents different temperature
        pressure and flowrate value
        
    """

    float_input=is_input_float([diam,flowrate,temp,pres])

    temp=pd.Series(temp)
    pres=pd.Series(pres)
    diam=pd.Series(diam)
    flowrate = pd.Series(flowrate)*1.667e-5

    idx = get_index([temp,pres,flowrate])
    
    D = particle_diffusivity(diam,temp,pres)

    rmuu = D.values*tubelength*(1./flowrate.values.reshape(-1,1))
    
    penetration = np.nan*np.ones(rmuu.shape)

    condition1 = (rmuu<0.009)
    condition2 = (rmuu>=0.009)

    penetration[condition1] = 1.-5.5*rmuu[condition1]**(2./3.)+3.77*rmuu[condition1]
    penetration[condition2] = 0.819*np.exp(-11.5*rmuu[condition2])+0.0975*np.exp(-70.1*rmuu[condition2])
    
    if float_input:
        return penetration[0][0]
    else:
        return pd.DataFrame(index=idx,columns=diam.values,data=penetration)

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

    dp : pandas.Series
        aerosol particle diameters
        unit: m

    Returns
    -------

    pandas.DataFrame
        Lung deposition fractions for alveoli ("DF_al"), trachea/bronchi ("DF_tb")
        head-airways ("DF_ha") and all combiend ("DF_tot")

    """

    # convert from meters to micrometers
    dp = dp.values*1e6

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
    dp = pd.Series(df.columns.values.astype(float))*1e6

    logdp = calc_bin_edges(pd.Series(dp)).values #array
    dlogdp = np.diff(logdp) #array

    # m2/cm-3 -> um2/cm-3
    surface_dist = surf_dist(df)*1e12 #dataframe

    # input needs ot be in m
    depo_fracs = calc_lung_df(dp*1e-6) #dataframe 

    ldsa_dist_al = surface_dist * depo_fracs.iloc[:,0].values.flatten() #dataframes
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

    tube_diam : float or series of lenght m
        Diameter of circular tube (m)
    flowrate : float or series of lenght n
        Volumetric flow rate (lpm)

    Returns
    -------

    float or dataframe of shape (n,m)
        Speed of fluid (m/s) 

    """

    float_input = is_input_float([tube_diam,flowrate])

    tube_diam = pd.Series(tube_diam)
    flowrate = pd.Series(flowrate)
 
    tube_diam = tube_diam.values
    flowrate = flowrate.values.reshape(-1,1)
    
    volu_flow = flowrate/60000.
    cross_area = np.pi*(tube_diam/2.)**2
    
    vel = volu_flow/cross_area

    if float_input:
        return vel[0][0] 
    else:
        return pd.DataFrame(index = flowrate.flatten(), columns = tube_diam, data = vel)

def pipe_reynolds(
    tube_diam,
    flowrate,
    temp=293.15,
    pres=101325.0):
    """
    Calculate Reynolds number in a tube

    Parameters
    ----------

    tube_diam : float or series of length m
        Inner diameter of the tube (m)
    flowrate : float or series of lenght n
        Volumetric flow rate (lpm)
    temp : float or series of length n
        Temperature in K
    pres : float or series of length n
        Pressure in Pa

    Returns
    -------

    float or dataframe of shape (n,m)
        Reynolds number

    """

    float_input = is_input_float([tube_diam,flowrate,temp,pres])

    tube_diam = pd.Series(tube_diam)
    flowrate = pd.Series(flowrate)
    temp = pd.Series(temp)
    pres = pd.Series(pres)

    idx = get_index([flowrate,temp,pres]) 

    tube_diam = tube_diam.values
    flowrate = flowrate.values.reshape(-1,1)
         
    volu_flow = flowrate/60000.
    visc = air_viscosity(temp)
    dens = air_density(temp,pres)

    visc = visc.values.reshape(-1,1)
    dens = dens.values.reshape(-1,1)

    Re = (dens*volu_flow*tube_diam)/(visc*np.pi*(tube_diam/2.0)**2)

    if float_input:
        return Re[0][0]
    else:
        return pd.DataFrame(index = idx, columns=tube_diam, data=Re)

def thab_dp2volts(thab_voltage,dp):
    """
    Convert particle diameters to DMA voltages

    Parameters
    ----------

    thab_voltage : float
        Voltage at THA+ peak (V)
    dp : float or series
        Particle diameters (nm)

    Returns
    -------

    float or series:
        DMA voltage (V) corresponding to dp

    Notes
    -----
    
    See https://doi.org/10.1016/j.jaerosci.2005.02.009

    Assumptions:

    1) Sheath flow is air
    2) Mobility standard used is THA+ monomer
    3) T = 293.15 K and p = 101325 Pa

    """

    thab_mob = (1.0/1.03)
        
    Zp = diam2mob(dp,293.15,101325.0,1)
   
    return (thab_voltage * thab_mob)/Zp


def thab_volts2dp(thab_voltage,dma_voltage):
    """
    Convert DMA voltages to particle diameters

    Parameters
    ----------

    thab_voltage : float
        Voltage at THA+ peak (V)
    dma_voltage : float
        DMA voltage (V)

    Returns
    -------

    float:
        particle diameter corresponding to DMA voltage (nm)

    Notes
    -----
    
    See https://doi.org/10.1016/j.jaerosci.2005.02.009

    Assumptions:

    1) Sheath flow is air
    2) Mobility standard used is THA+ monomer
    3) T = 293.15 K and p = 101325 Pa

    """

    thab_mob = (1.0/1.03)
        
    Zp = (thab_voltage*thab_mob)/dma_voltage

    dp = mob2diam(Zp,293.15,101325.0,1)
    
    return dp

def eq_charge_frac(dp,N):
    """
    Calculate equilibrium charge fraction using Wiedensohler (1988) approximation

    Parameters
    ----------

    dp : float
        Particle diameter (m)
    N : int
        Amount of elementary charge in range [-2,2]

    Returns
    -------

    float
        Fraction of particles of diameter dp having N 
        elementary charges 

    """

    a = {-2:np.array([-26.3328,35.9044,-21.4608,7.0867,-1.3088,0.1051]),
        -1:np.array([-2.3197,0.6175,0.6201,-0.1105,-0.1260,0.0297]),
        0:np.array([-0.0003,-0.1014,0.3073,-0.3372,0.1023,-0.0105]),
        1:np.array([-2.3484,0.6044,0.4800,0.0013,-0.1544,0.0320]),
        2:np.array([-44.4756,79.3772,-62.8900,26.4492,-5.7480,0.5059])}

    if (np.abs(N)>2):
        raise Exception("Number of elementary charges must be 2 or less")
    elif ((dp<20e-9) & (np.abs(N)==2)):
        return 0
    else:
        return 10**np.sum(a[N]*(np.log10(dp*1e9)**np.arange(6)))

def utc2solar(utc_time,lon,lat):
    """  
    Convert utc time to solar time (solar maximum occurs at noon)

    Parameters
    ----------

    utc_time : pandas Timestamp
    lon : float
        Location's longitude
    lat : float
        Location's latitude

    Returns
    -------

    pandas Timestamp
        solar time

    """

    # Create observer based on location
    observer = Observer(latitude=lat,longitude=lon)

    date = pd.to_datetime(utc_time.strftime("%Y-%m-%d"))

    # Convert time objects to float
    utc_time_num = dts.date2num(utc_time)
    noon_utc_time_num = dts.date2num(pd.to_datetime(noon(observer, date=date))) 
    noon_solar_time_num = dts.date2num(pd.to_datetime(date + pd.Timedelta("12 hours")))

    # Convert utc to solar time
    solar_time_num = (utc_time_num * noon_solar_time_num) / noon_utc_time_num

    solar_time = pd.to_datetime(dts.num2date(solar_time_num)).tz_convert(None)
    
    return solar_time




def calc_mob_ratio(neg_ions,pos_ions):
    
    neg_ions[neg_ions<=0] = np.nan
    pos_ions[pos_ions<=0] = np.nan

    neg_ions = neg_ions.interpolate(limit_direction="both",axis=1).rolling(window=5).median().interpolate(limit_direction="both")
    pos_ions = pos_ions.interpolate(limit_direction="both",axis=1).rolling(window=5).median().interpolate(limit_direction="both")

    x = np.exp(np.log(neg_ions.values/pos_ions.values)/2.0)

    return pd.DataFrame(index = neg_ions.index, columns=neg_ions.columns, data=x)

def atmo_ion_frac(dp,q,temp=273.15,mob_ratio=1.0):

    if (np.abs(q)==1):
        alpha = 0.9630*np.exp(7.6019/(dp+2.2476))
    elif (np.abs(q)==2):
        alpha = 0.9826+0.9435*np.exp(-0.0478*dp)
    else:
        alpha = 1.0
   
    x = mob_ratio
    T = temp

    f = (E/np.sqrt(4*np.pi**2*E_0*alpha*dp*K_B*T)*
            np.exp( 
                (-(q-(2*np.pi*E_0*alpha*dp*K_B*T)/(E**2)*np.log(x))**2)/
                ((4*np.pi*E_0*alpha*dp*K_B*T)/(E**2)) 
            )) 
    
    return f


def ions2particles(neg_ions,pos_ions,temp=293.15,mob_ratio=1.0):
    """
    Estimate particle number size distribution from ions using Li et al. (2022)

    Parameters
    ----------

    neg_ions : pandas dataframe of shape (n,m)
        negative ion number size distribution
    pos_ions : pandas dataframe of shape (n,m)
        positive ion number size distribution
    temp : float or series of length n
        ambient temperature in K
    mob_ratio : float
        mobility ratio to be used
        default 1.0
    
    Returns
    -------

    pandas dataframe of shape (n,m)
        estimated particle number size distribution

    References
    ----------

    Li et al. (2022), https://doi.org/10.1080/02786826.2022.2060795

    """

    # Template for the particle number size distribution  
    particles = np.nan*np.ones(neg_ions.shape)

    dp = neg_ions.columns.values.astype(float)

    # Calculate the alpha matrix (q,dp) -> alpha
    alpha = np.ones((5,neg_ions.shape[1]))
    q = np.array([1,2,3,4,5]).reshape(-1,1)
    for i in range(5):
        if (i==0):
            alpha[i,:] = 0.9630*np.exp(7.6019/(dp+2.2476))
        elif (i==1):
            alpha[i,:] = 0.9826+0.9435*np.exp(-0.0478*dp)
        else:
            alpha[i,:] = 1.0
   
    if isinstance(temp,float):
        temp = pd.Series(index = neg_ions.index, data=temp)

    if mob_ratio is None:
        X = calc_mob_ratio(neg_ions,pos_ions)

    # For each measurement time calculate the particle number size distribution
    for i in range(neg_ions.shape[0]):

        if mob_ratio is not None:
            x = mob_ratio
        else:
            x = X.values[i,:]
        T = temp.values[i]

        # Calculate the positive and negative charge fractions
        f_pos = (E/np.sqrt(4*np.pi**2*E_0*alpha*dp*K_B*T)*
            np.exp( 
                (-(q-(2*np.pi*E_0*alpha*dp*K_B*T)/(E**2)*np.log(x))**2)/
                ((4*np.pi*E_0*alpha*dp*K_B*T)/(E**2)) 
            )) 
        f_neg = (E/np.sqrt(4*np.pi**2*E_0*alpha*dp*K_B*T)*
            np.exp( 
                (-(-q-(2*np.pi*E_0*alpha*dp*K_B*T)/(E**2)*np.log(x))**2)/
                ((4*np.pi*E_0*alpha*dp*K_B*T)/(E**2)) 
            ))
        # Add the charge fractions together 
        f_tot = f_pos + f_neg
        f = np.nansum(f_tot,axis=0)
        f[f<=0]=np.nan
        # Calculate the particles
        particles[i,:] = (pos_ions.values[i,:] + neg_ions.values[i,:])/f

    return pd.DataFrame(data=particles,index = neg_ions.index,columns=neg_ions.columns)

def calc_tube_residence_time(tube_diam,tube_length,flowrate):
    """
    Calculate residence time in a circular tube

    Parameters
    ----------

    tube_diam : float or series of length m
        Inner diameter of the tube (m)
    tube_length : float or series of length m
        Length of the tube (m)
    flowrate : float or series of length n
        Volumetric flow rate (lpm)

    Returns
    -------

    float or dataframe of shape (n,m)
        Average residence time in seconds

    """

    float_input = is_input_float([tube_diam,tube_length,flowrate])

    tube_diam = pd.Series(tube_diam)
    tube_length = pd.Series(tube_length)
    flowrate = pd.Series(flowrate)

    tube_diam = tube_diam.values
    tube_length = tube_length.values
    flowrate = flowrate.values.reshape(-1,1)
         
    volu_flow = flowrate/60000.
    tube_volume = np.pi*tube_diam**2*(1/4.)*tube_length

    rt = tube_volume/volu_flow

    if float_input:
        return rt[0][0]
    else:
        return pd.DataFrame(index = flowrate.flatten(), columns=tube_volume, data=rt)

def calc_ion_production_rate(
    df_ions,
    df_particles,
    temp=293.15,
    pres=101325.0):
    """
    Calculate the ion production rate from measurements

    Parameters
    ----------

    df_ions : dataframe of shape (n,m)
        negative or positive ion number size distribution unit cm-3
    df_particles : dataframe of shape (n,m)
        particle number size distribution unit cm-3
    temp : float or series of length n
        ambient temperature unit K
    pres : float or series of length n
        ambient pressure unit Pa

    Returns
    -------

    series of lenght n
        ion production rate in cm-3 s-1

    Notes
    -----

    """

    temp = pd.Series(temp)
    pres = pd.Series(pres)

    if len(temp)==1:
        temp = pd.Series(index = df_ions.index, data = temp)
    else:
        temp = temp.reindex(df_ions.index, method="nearest")

    if len(pres)==1:
        pres = pd.Series(index = df_ions.index, data = pres)
    else:
        pres = pres.reindex(df_ions.index, method="nearest")

    alpha = 1.6e-6 # cm3 s-1
    dp1 = 1e-9
    dp2 = 2e-9
    
    cluster_ion_conc = calc_conc(df_ions,dp1,dp2)
    
    ion_dp = df_ions.columns.values.astype(float)
    
    findex = np.argwhere((ion_dp>dp1) & (ion_dp<dp2)).flatten() 

    sink_term = np.zeros(df_ions.shape[0])
    for dp in ion_dp[findex]: 
        sink_term = sink_term + calc_coags(df_particles,dp,temp,pres).values.flatten()*cluster_ion_conc.values.flatten()
    
    # Ion-ion recombination term
    rec_term = cluster_ion_conc.values.flatten()**2*alpha

    ion_production_rate = rec_term + sink_term

    return pd.Series(index=df_ions.index, data=ion_production_rate)

def dma_volts2mob(Q,R1,R2,L,V):
    """
    Theoretical selected mobility from cylindrical DMA

    Parameters
    ----------

    Q : float
        sheath flow rate, unit lpm

    R1 : float
        inner electrode radius, unit m

    R2 : float
        outer electrode radius, unit m

    L : float
        effective electrode length, unit m

    V : float or series
        applied voltage, unit V

    Returns
    -------

    float or series
        selected mobility, unit cm2 s-1 V-1

    """

    return ((Q*1.667e-5)*np.log(R2/R1))/(2.*np.pi*L*V)*1e4

def dma_mob2volts(Q,R1,R2,L,Z):
    """
    Cylindrical DMA voltage corresponding to mobility

    Parameters
    ----------

    Q : float
        sheath flow rate, unit lpm

    R1 : float
        inner electrode radius, unit m

    R2 : float
        outer electrode radius, unit m

    L : float
        effective electrode length, unit m

    Z : float
        mobility, unit cm2 s-1 V-1

    Returns
    -------

    float
        DMA voltage, unit V

    """

    return ((Q*1.667e-5)*np.log(R2/R1))/(2.*np.pi*L*Z*1e-4)



def conical_dma_mob2volts(Q, R1_max, R2, L, alpha, Z):
    """
    Conical DMA voltage corresponding to mobility

    Parameters
    ----------

    Q : float
        sheath flow rate, unit lpm

    R1_max : float
        inner electrode radius at outlet at distance L, unit m

    R2 : float
        outer electrode radius, unit m

    L : float
        effective electrode length, unit m
        
    alpha : float
        tapering angle, unit degrees

    Z : float
        mobility, unit cm2 s-1 V-1

    Returns
    -------

    float
        DMA voltage, unit V

    """

    # Convert to radians
    alpha = alpha*(np.pi/180.)
    
    # Calculate the geometric factor K_T
    x = np.linspace(0,L,100)
    Rx = R1_max - (L-x) * np.tan(alpha)
    K_T = np.log(R2/R1_max)/L * np.trapz(y = 1./np.log(R2/Rx), x = x)
    
    # Calculate the voltage
    V = ((Q*1.667e-5)*np.log(R2/R1_max))/(2*np.pi*L*(Z*1e-4)*K_T)
    
    return V


def tubeloss_turbulent(diam, flowrate, tube_length, tube_diam, temp=293.15, pres=101325.):
    """
    Calculate particle losses to walls of a straight cylindrical 
    tube assuming a turbulent flow regime and air as the carrier gas.

    Parameters
    ----------
    
    diam : float or series of length m
        Particle diameters for which to calculate the
        losses, unit: m
    flowrate : float or series of length n
        unit: L/min
    tube_length : float
        Length of the cylindrical tube
        unit: m
    tube_diam : float
        Diameter of the cylindrical tube
        unit: m
    temp : float or series of length n
        temperature
        unit: K
    pres : float or series of lenght n
        air pressure
        unit: Pa

    Returns
    -------

    float or dataframe of shape (n,m)
        Fraction of particles passing through.
        Each column represents diameter and each
        each row represents different temperature
        pressure and flowrate value

    """
    
    float_input=is_input_float([diam,flowrate,temp,pres])

    temp=pd.Series(temp)
    pres=pd.Series(pres)
    diam=pd.Series(diam)
    flowrate = pd.Series(flowrate)*1.667e-5

    idx = get_index([temp,pres,flowrate])
    
    # Average flow velocity
    flow_velo = flow_velocity_in_pipe(tube_diam, flowrate) # shape: (len(flowrate),len(tube_diam)) or float
    flow_velo = flow_velo.values.flatten()

    # Reynolds number
    Re = pipe_reynolds(tube_diam, flowrate, temp, pres) # shape: (maxlen(flowrate,temp,pres),len(tube_diam)) or float
    Re = Re.values.flatten()

    # Particle diffusivity
    D = particle_diffusivity(diam,temp,pres) # shape: (maxlen(temp,pres),len(diam)) or float
    D = D.values

    # Air density
    air_dens = air_density(temp,pres) # shape: maxlen(temp,pres) or float
    air_dens = air_dens.values.flatten()

    # Air viscosity
    air_visc = air_viscosity(temp) # shape: len(temp) or float
    air_visc = air_visc.values.flatten()

    # Diffusive deposition velocity
    #V_d = ((0.04*flow_velo)/(Re**(1/4.)))*((air_dens*D)/(air_visc))**(2/3.)

    delta = (28.5*tube_diam*D**(1/4.))/(Re**(7/8.)*(air_visc/air_dens)**1/4.) 
    V_d = D/delta

    penetration = np.exp(-(4*V_d*tube_length)/(tube_diam * flow_velo))

    if float_input:
        return penetration[0][0]
    else:
        return pd.DataFrame(index=idx,columns=diam.values,data=penetration)

def sample_from_dist(x,y,n):
    """
    Draw n samples from empirical distribution defined by points (x,y)

    Parameters
    ----------

    x : numpy array
        x-data points

    y : numpy array
        y-data points

    n : int
        number of samples to draw

    Returns
    -------

    numpy array
        samples drawn

    """

    dist = y/np.sum(y)
    cdf = np.cumsum(dist)
    cdf = cdf / cdf[-1]
    random_values = np.random.rand(n)
    samples = np.interp(random_values, cdf, x)

    return samples


def denan(df):
    """
    Interpolate away any nans and drop nan tails

    Parameters
    ----------

    df : pandas datafarme

    Returns
    -------

    pandas dataframe

    """
    return df.interpolate(limit_area="inside").dropna(how="all",axis=0)

def nanoranking(df, dmin, dmax, row_threshold=0, col_threshold=0):
    """
    Simplified method of calculating the nanorank

    Parameters
    ----------

    df : pandas dataframe
        number size distribution
    dmin : float
        lower diameter limit
    dmax : float
        upper diameter limit
    row_threshold : float
        maximum fraction of nans when calculating the number concentartion
    col_threshold : float
        maximum fraction of nans in the conentration timeseries
    
    Returns
    -------

    dictionary
        the result dictionary has the following keys:
        
        `norm_conc`: Concentration with removed background

        `rank`: Value of the peak normalized concentration

        `rank_time`: Time where the peak occurs

    Notes
    -----

    The nanorank is calculated for one day day. See Aliaga et al 2023  

    """

    # Calculate the number concentration
    conc = calc_conc_interp(df,dmin,dmax, threshold = row_threshold)

    # Check if rows has enough data points
    conc = filter_nans(conc, threshold = col_threshold, axis=0)

    if conc.empty:
        return None

    # Subtract the background
    norm_conc = conc.iloc[:,0]-np.nanmedian(conc)
    # Retrieve the rank and the rank time
    rank = norm_conc.max()
    rank_time = norm_conc.idxmax()

    result = {
        "norm_conc": norm_conc, # series
        "rank": rank,
        "rank_time": rank_time
    }

    return result

def nanoranking_conc(conc, nan_threshold=0, include_concs=False):
    """
    Simplified method of calculating the nanorank

    Parameters
    ----------

    conc : pandas series
        number concentration time series in the diameter range of interst
    nan_threshold : float
        maximum fraction of nans when calculating the number concentartion for each day
    include_concs : boolean
        Include the background removed concentrations for each day in the results or not
    
    Returns
    -------

    dictionary
        the result dictionary has the following keys:
        
        `norm_conc`: Concentration with removed background (if include_concs = True)

        `rank`: Value of the peak normalized concentration

        `rank_time`: Time where the peak occurs

    Notes
    -----

    The nanorank is calculated for one day day. See Aliaga et al 2023  

    """

    days = conc.index.strftime("%Y%m%d").unique()

    results = pd.DataFrame(index=days,columns=["rank","rank_time"])

    for d in days:
        conc_d = conc.iloc[conc.index.strftime("%Y%m%d")==d].to_frame()

        # Check if there are enough data points
        conc_d_filt = filter_nans(conc_d, threshold = nan_threshold, axis=0)

        if conc_d_filt.empty:
            results.loc[d] = [np.nan,np.nan]
            print(d,None)
            continue

        # Subtract the background
        norm_conc = conc_d_filt.iloc[:,0]-np.nanmedian(conc_d_filt)
        
        # Retrieve the rank and the rank time
        rank = norm_conc.max()
        rank_time = norm_conc.idxmax()

        results.loc[d] = [rank, rank_time] 
        
        print(d,rank)

    results.index = pd.to_datetime(results.index)
    
    return results


def normalize_signal(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

def zscore(a):
    a = np.asarray(a, dtype=float)
    return (a - a.mean()) / a.std(ddof=0)

def normalized_cross_correlation(x, y, gamma=0.25):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)

    xzm = zscore(x)
    yzm = zscore(y)
    
    # numerator
    corr = correlate(xzm, yzm, mode='full')

    # normalized correlation
    lags = correlation_lags(nx, ny, mode='full')
    
    N_k = nx-np.abs(lags)

    norm = (N_k)**gamma

    norm_corr = corr/norm

    return lags, norm_corr


def cross_corr_gr(
    df, 
    dmin, 
    dmax, 
    smoothing_window=None,
    tau_limit=12.0, 
    number_of_divisions=1, 
    nan_threshold=0.0, 
    data_reso=1.0,
    gamma=0.25,
    verbose=False):
    """
    Calculate GR using cross-correlation method

    Parameters
    ----------

    df : pandas dataframe
        Aerosol number size distribution
    dmin : float
        Lower size limit for GR
    dmax : float
        Upper size limit for GR
    smoothing_window : float
        Window length used in smoothing the data in hours
        If `None`, no smoothing is applied
    tau_limit : float
        Range of time lags used in hours
    number_of_divisions : int
        The number of divisions applied to the size range 
    row_threshold : float
        Maximum fraction of NaNs present in the rows
    col_thershold : float
        Maximum fraction of NaNs present in the columns
    data_reso : float
        Data resolution in hours
    verbose : boolean
        If True then GR, lags and cross correlations are returned
        If False then only GR is returned

    Returns
    -------

    dictionary
        Results in a dictionary:

        If verbose is True

        `gr`: total growth rate in nm/h
        `gr_incr`: growth rates in each increment
        `tau_max_incr`: tau_max in each increment
        `lag`: lags in seconds for each increment
        `corr`: cross correlations for each increment
        `t`: interpolated timestamps (seconds)
        `n1`: Interpolated unnormalized concs in lower channels
        `n2`: Interpolated unnormalized concs in upper channels


        If verbose is False
        `gr`: total growth rate in nm/h


    """ 
    
    diams = df.columns.astype(float).values

    if ((dmin<np.min(diams)) | (dmax>np.max(diams))):
        return -999
    
    # Find indices where values are in range (low, high)
    in_range_idx = np.where((diams > dmin) & (diams < dmax))[0]

    num_size_channels = len(in_range_idx)

    if (num_size_channels<number_of_divisions):
        print("Number of size channels is less han number of size range divisions")
        return -999

    # Include 1 around each index
    expanded_idx = np.unique(np.concatenate([in_range_idx - 1, in_range_idx, in_range_idx + 1]))

    # Clip indices to stay within bounds of array
    expanded_idx = expanded_idx[(expanded_idx >= 0) & (expanded_idx < len(diams))]

    df_in_range = df.iloc[:,expanded_idx]

    if np.any((df_in_range.isnull().mean(axis=0)>=nan_threshold).values):
        return -999
    #if np.any((df_in_range.isnull().mean(axis=1)>=col_threshold).values):
    #    return -999

    if (df.index.is_monotonic_increasing==False):
        return -999

    if ((df.shape[0]/data_reso) < (23.0/data_reso)):
        return -999

    # Filter out the nans from the whole data frame
    df = denan(df)

    if smoothing_window is not None:
        window_length = int(np.round(smoothing_window/data_reso))
        df = df.rolling(window=window_length, min_periods=1, center=True).mean()

    logdp = np.log10(diams)

    # Find the diameters 
    step_size = (np.log10(dmax)-np.log10(dmin))/number_of_divisions
    d = dmin 
    conc_cols = [d]
    for i in range(number_of_divisions):
        d = 10**(np.log10(d) + step_size)
        conc_cols.append(d)
    
    # Interpolate to the new diameters
    dp_grid = np.log10(conc_cols)
    data = np.nan*np.ones((df.shape[0],len(dp_grid)))
    for j in range(df.shape[0]):
        data[j,:] = np.interp(dp_grid,logdp,df.iloc[j,:].values)

    # Interpolate to new times
    t = (df.index-pd.to_datetime(df.index[0].strftime("%Y-%m-%d"))).total_seconds().values
    t_grid = np.arange(0,60*60*24+1)
 
    data2 = np.nan*np.ones((len(t_grid),len(dp_grid)))
    for j in range(len(dp_grid)):
        data2[:,j] = np.interp(t_grid,t,data[:,j])

   # d1 = dmin
   # d2 = 10**(np.log10(dmin) + step_size)
    tau_max_tot = 0
    corr_max_tot = 0
    max_index = 0

    all_gr = []
    all_tau_max = []
    all_corrs = []
    all_lags = []
    all_channel1 = []
    all_channel2 = []

    for i in range(number_of_divisions):

        max_index = i
        channel1 = data2[:,i].flatten()
        channel2 = data2[:,i+1].flatten()
        
        lag,corr = normalized_cross_correlation(channel1,channel2,gamma=gamma)
  
        start = int((60*60*24) - tau_limit*60*60 - 1)
        end = int((60*60*24) + tau_limit*60*60 - 1)

        lag = lag[start:end]
        corr = corr[start:end]

        tau_max = -lag[np.argmax(corr)] # seconds
        #corr_max = np.max(corr)
        max_lag = np.max(-lag)

        # sanity check the tau_max
        # If we have majority of size increments 
        if ((tau_max>0)&(tau_max<max_lag)):
            tau_max_tot += tau_max
            all_gr.append((conc_cols[max_index+1]-dmin)*1e9/((tau_max)/(60*60)))
            all_tau_max.append(tau_max)
            all_channel1.append(channel1)
            all_channel2.append(channel2)
            all_lags.append(-lag)
            all_corrs.append(corr)
            #corr_max_tot += corr_max
        elif (tau_max==0):
            return -888
        elif (tau_max<0):
            return -777
        elif (tau_max==max_lag):
            return -666
        else:
            return -555

    gr = (conc_cols[max_index+1]-dmin)*1e9/((tau_max_tot)/(60*60)) #nm h-1

    print(gr)

    if verbose:
        return {"gr":gr,"gr_incr":all_gr,"tau_max_incr":all_tau_max,"lag":all_lags,"corr":all_corrs,"t":t_grid,"n1":all_channel1,"n2":all_channel2}
    else:
        return {"gr":gr}


