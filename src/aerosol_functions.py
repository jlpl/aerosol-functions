"""
Collection of functions to analyze atmospheric aerosol data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dts
from matplotlib.ticker import LogLocator
from matplotlib import colors
from datetime import datetime, timedelta, timezone
from scipy.optimize import minimize
from scipy.integrate import trapezoid

def datenum2datetime(datenum,tz=None):
    """
    Convert from matlab datenum to python datetime 

    Parameters
    ----------

    datenum : `float`
        A serial date number representing the whole and 
        fractional number of days from 1-Jan-0000 to a 
        specific date (MATLAB datenum)
  
    tz : `int` or `None`
        Timezone offset in minutes from UTC
        `None` implies timezone unaware

    Returns
    -------

    `pandas.Timestamp`

    """
    dt = (datetime.fromordinal(int(datenum)) + timedelta(days=datenum%1) - timedelta(days = 366))
    if tz is not None:
        tz_offset = timezone(timedelta(minutes=tz))
        dt = dt.replace(tzinfo=tz_offset)

    return pd.to_datetime(dt.isoformat())

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

def bin_df(df, t_min, t_max, reso, q=0.5):
    """ Utility function for binning timeseries data

    Parameters
    ----------

    df : pandas.DataFrame
        Aerosol number size distribution

        `df.index` time  
        `df.columns` particle diameter (m)
        `df.values` normalized concentrations (dN/dlogDp) 

    t_min : datetime or str
        first bin lower limit

    t_max : datetime or str
        last bin upper limit

    reso : int or str
        desired time resolution in minutes
        or pandas time offset alias 

    q : float
        quintile of data calculated per bin

        default is the median (0.5)

    Returns
    -------

    pandas.DataFrame
        Binned aerosol number size distribution

        All bins have constant width determined by reso and they
        share edges. If a bin has no values it is given a value of `NaN`

    """
    if isinstance(reso,int):
        reso = pd.Timedelta(minutes=reso)
    if isinstance(reso,str):
        pass

    ix = pd.date_range(t_min,t_max,freq=reso)
    half_step = (ix[1] - ix[0])/2.
   
    data = []
    index = []

    for i in range(len(ix)-1):
        df_block = df.iloc[((df.index>=ix[i]) & (df.index<ix[i+1])),:].median().values.flatten()
        if len(df_block)==0:
            df_block = np.nan*np.ones(len(df.columns))
        data.append(df_block)
        index.append(ix[i] + half_step)

    return pd.DataFrame(index = index, data = data, columns = df.columns)

def generate_log_ticks(min_exp,max_exp):
    """
    Generate ticks and ticklabels for log axis

    Parameters:
    -----------
    
    min_exp : int
        The exponent in the smallest power of ten

    max_exp : int
        The exponent in the largest power of ten

    Returns:
    --------

    numpy.array
        tick values

    list of str
        tick labels for each power of ten

    """
    x=np.arange(1,10)
    y=np.arange(min_exp,max_exp).astype(float)
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
    v,
    ax=None,
    vmin=10,
    vmax=100000,
    time_reso=2,
    time_formatter="%H:%M"):    
    """ 
    Plot aerosol particle number-size distribution surface plot

    Parameters
    ----------

    v : pandas.DataFrame
        Aerosol number size distribution

        time (index) should be have constant resolution, 
        otherwise the time axis will not be correct

    ax : axes object
        axis on which to plot the data
        if `None` the axis are created

    vmin : float or int
        color scale lower limit

    vmax : float or int
        color scale upper limit

    time_reso : `int`
        Time resolution of ticks given in hours

    time_formatter : `str`
        Define the format of time ticklabels
        
    """

    if ax is None:
        fig,handle = plt.subplots(figsize=(10,4))
    else:
        handle=ax

    dp = v.columns.values.astype(float)
    dndlogdp = v.values.astype(float)

    log_ticks,log_tick_labels = generate_log_ticks(-10,-4)

    norm = colors.LogNorm(vmin=vmin,vmax=vmax)
    color_ticks = LogLocator(subs=range(10))

    handle.set_yticks(log_ticks)
    handle.set_yticklabels(log_tick_labels)

    if v.index[0].utcoffset() is None:
        t1=dts.date2num(v.index[0])+v.index[0].second/(60.*60.*24.)
        t2=dts.date2num(v.index[-1])+v.index[-1].second/(60.*60.*24.)
    else:
        t1=dts.date2num(v.index[0])+v.index[0].utcoffset().seconds/(60.*60.*24.)
        t2=dts.date2num(v.index[-1])+v.index[-1].utcoffset().seconds/(60.*60.*24.)

    dp1=np.log10(dp.min())
    dp2=np.log10(dp.max())

    img = handle.imshow(
        np.flipud(dndlogdp.T),
        origin="upper",
        aspect="auto",
        cmap="turbo",
        norm=norm,
        extent=(t1,t2,dp1,dp2)
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

def dndlogdp2dn(df):
    """    
    Convert from normalized number concentrations to
    unnormalized number concentrations assuming that 
    the size channels have common edges.

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
    Calculate air viscosity
    using Enskog-Chapman theory

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

def calc_coags(df,Dp,dp,temp,pres):
    """ 
    Calculate coagulation sink

    Kulmala et al (2012): doi:10.1038/nprot.2012.091 

    Parameters
    ----------

    df : pandas.DataFrame
        Aerosol number size distribution

    Dp : float
        Particle diameter for which you want to calculate the CoagS, 
        unit: m

    temp : pandas.DataFrame
        Ambient temperature timeseries
        unit: K

    pres : pandas.DataFrame
        Ambient pressure timeseries
        unit: Pa

    Returns
    -------
    
    pandas.DataFrame
        Coagulation sink time series,
        unit: s-1

    """

    df = df.loc[:,df.columns>=Dp]
    temp = temp.reindex(df.index, method="nearest")
    pres = pres.reindex(df.index, method="nearest")
    
    a = dndlogdp2dn(df)
    b = 1e6*coagulation_coef(Dp,df.columns.values,temp.values,pres.values)
 
    return (a*b).sum(axis=1,min_count=1)
   
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

    float or numpy.array
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

    temp : pandas.DataFrame or float
        Ambient temperature corresponding to the data, unit: K
        If single value given it is used for all data

    pres : pandas.DataFrame or float
        Ambient pressure corresponding to the data, unit: Pa
        If single value given it is used for all data

    Returns
    -------
    
    pandas.DataFrame
        condensation sink time series, unit: s-1

    """
    
    if isinstance(temp,float):
        temp = pd.DataFrame(index = df.index, columns=["Temperature"], data=temp)
    else:
        temp = temp.reindex(df.index, method="nearest")

    if isinstance(pres,float):
        pres = pd.DataFrame(index = df.index, columns=["Pressure"], data=pres)
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

    cs = (4.*np.pi*diffu.flatten())*df2.values

    return pd.DataFrame(index=df.index,columns=["CS"],data=cs)

def calc_conc(df,dmin,dmax):
    """
    Calculate particle number concentration from aerosol 
    number-size distribution

    Parameters
    ----------

    df : pandas.DataFrame
        Aerosol number-size distribution

    dmin : float
        Size range lower diameter, unit: m

    dmax : float
        Size range upper diameter, unit: m

    Returns
    -------
    
    pandas.DataFrame
        Number concentration in the given size range, unit: cm-3

    """

    dp = df.columns.values.astype(float)
    findex=np.argwhere((dp<=dmax)&(dp>=dmin)).flatten()
    if len(findex)==0:
        return np.nan*np.ones(df.shape[0])
    dp=dp[findex]
    conc=df.iloc[:,findex]
    logdp_mid=np.log10(dp)
    logdp=(logdp_mid[:-1]+logdp_mid[1:])/2.0
    logdp=np.append(logdp,logdp_mid.max()+(logdp_mid.max()-logdp.max()))
    logdp=np.insert(logdp,0,logdp_mid.min()-(logdp.min()-logdp_mid.min()))
    dlogdp=np.diff(logdp)
    conc = np.nansum(conc*dlogdp,axis=1)
    return pd.DataFrame(index=df.index,data=conc,columns=["conc"])

#    dp = df.columns.values.astype(float)
#    df2 = df.loc[:,((dp>=dmin) & (dp<=dmax))]
#    dp2 = df2.columns.values.astype(float)
#    df2.columns = np.log10(dp2)
#    dense_dp = np.full(len(df2.columns)*10,np.nan)[:-9]
#    dense_dp[::10] = df2.columns.values
#    dense_dp = pd.DataFrame(dense_dp).interpolate().values.flatten()
#    df3 = df2.reindex(dense_dp,axis=1)
#    df4 = df3.interpolate(axis=1)
#    conc = df4.apply(trapezoid,args=(df4.columns,),axis=1) 
#    return pd.DataFrame(index = df.index, data = conc)

def calc_formation_rate(dp1,dp2,conc,coags,gr):
    """
    Calculate particle formation rate

    Kulmala et al (2012): doi:10.1038/nprot.2012.091

    Parameters
    ----------
    
    dp1 : float
        Lower diameter of the size range, unit: m

    dp2 : float
        Upper diameter of the size range, unit m

    conc : pandas.DataFrame
        particle number concentration timeseries
        in the size range dp1...dp2, unit cm-3

    coags : pandas.DataFrame
        Coagulation sink timeseries for particles 
        in the size range dp1...dp2. unit s-1 

        Usually approximated as coagulation sink for particle size dp1, 
        unit s-1

    gr : float
        Growth rate for particles out of the size range dp1...dp2, 
        unit nm h-1

    Returns
    -------

    pandas.DataFrame
        particle formation rate for diameter dp1, unit: cm3 s-1

    """
   
    # Fit the coags to the conc index
    coags = coags.reindex(conc.index,method="nearest")
    time_diff = pd.DataFrame(conc.index).diff().astype('datetime64[s]').values
    conc_term = np.diff(conc.values)/time_diff
    sink_term = (coags.values[1:] + coags.values[:-1])/2. * (conc.values[1:] + conc.values[:-1])/2.
    gr_term = (2.778e-13*gr)/(dp2-dp1) * (conc.values[1:] + conc.values[:-1])/2.
    formation_rate = conc_term + sink_term + gr_term

    return pd.DataFrame(index = conc.index, data = formation_rate)

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

    time : numpy.array of datetime objects
        Time associated with the measurements 

    dp1 : float
        Lower diameter of the size range, unit: m

    dp2 : float
        Upper diameter of the size range, unit: m

    conc_pos : numpy.array
        Positive ion number concentration in the size range dp1...dp2, unit: cm-3

    conc_neg : numpy.array
        Negative ion number concentration in the size range dp1...dp2, unit: cm-3

    conc_pos_small : numpy.array
        Positive ion number concentration for ions smaller than dp1, unit: cm-3

    conc_neg_small : numpy.array
        Negative ion number concentration for ions smaller than dp1, unit: cm-3

    conc : numpy.array
        Particle number concentration in the size range dp1...dp2, unit: cm-3

    coags : numpy.array
        Coagulation sink for particles in the size range dp1...dp2.
        Usually approximated as coagulation sink for particle size dp1, 
        unit: s-1

    gr : float
        Growth rate for particles out of the size range dp1...dp2, unit: nm h-1

    Returns
    -------

    pandas.DataFrame
        Time as index
        Negative ion formation rate for diameter dp1, unit : cm3 s-1

    pandas.DataFrame
        Time as index
        Positive ion formation rate for diameter dp1, unit: cm3 s-1

    """

    # Reindex everything to conc_neg
    coags = coags.reindex(conc_neg.index,method="nearest")
    conc_pos = conc_pos.reindex(conc_neg.index,method="nearest")
    conc = conc.reindex(conc_neg.index,method="nearest")
    conc_neg_small = conc_neg_small.reindex(conc_neg.index,method="nearest")
    conc_pos_small = conc_pos_small.reindex(conc_neg.index,method="nearest")

    time_diff=pd.DataFrame(conc_neg.index).diff().astype('datetime64[s]').values

    alpha = 1.6e-6 # cm3 s-1
    Xi = 0.01e-6 # cm3 s-1

    coags = (coags[1:] + coags[:-1])/2.
    conc_pos = (conc_pos[1:] + conc_pos[:-1])/2.
    conc_neg = (conc_neg[1:] + conc_neg[:-1])/2.
    conc_pos_small = (conc_pos_small[1:] + conc_pos_small[:-1])/2.
    conc_neg_small = (conc_neg_small[1:] + conc_neg_small[:-1])/2.
    conc = (conc[1:] + conc[:-1])/2.

    pos_conc_term = np.diff(conc_pos)/time_diff
    pos_sink_term = coags * conc_pos
    pos_gr_term = (2.778e-13*gr)/(dp2-dp1) * conc_pos
    pos_recombination_term = alpha * conc_pos * conc_neg_small
    pos_charging_term = Xi * conc * conc_pos_small
    pos_formation_rate = pos_conc_term + pos_sink_term + pos_gr_term + pos_recombination_term - pos_charging_term

    neg_conc_term = np.diff(conc_neg)/time_diff
    neg_sink_term = coags * conc_neg
    neg_gr_term = (2.778e-13*gr)/(dp2-dp1) * conc_neg
    neg_recombination_term = alpha * conc_neg * conc_pos_small
    neg_charging_term = Xi * conc * conc_neg_small
    neg_formation_rate = neg_conc_term + neg_sink_term + neg_gr_term + neg_recombination_term - neg_charging_term

    neg_J = pd.DataFrame(index = time, data = neg_formation_rate)
    pos_J = pd.DataFrame(index = time, data = pos_formation_rate)
 
    return neg_J, pos_J

def stack_data(filenames,t_min,t_max,reso):
    """
    Concatenate data

    If multiple data tables share the same header
    and have time as index then this function can be
    used to concatenate the data into a single large 
    data frame.
    
    Parameters
    ----------
    
    filenames : list of strings
        filenames of the data to be combined

    t_min : datetime or str

    t_max : datetime or str

    reso : int or str
        desired time resolution of the output in minutes
        or pandas time offset alias
    
    Returns
    -------

    pandas.DataFrame
        stacked data

    """

    df_combined = pd.DataFrame([])
    for f in filenames:
        print(f)
        df = pd.read_csv(f,index_col=0,parse_dates=True)
        df_combined = pd.concat((df_combined,df),axis=0)

    if len(df_combined)==0:
        return df_combined
    else:
        df_combined = df_combined.sort_index()
        df_combined = bin_df(df_combined,t_min,t_max,reso)

        return df_combined
