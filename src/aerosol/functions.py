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

def generate_log_ticks(min_exp,max_exp):
    """
    Generate ticks and ticklabels for log axis

    Parameters
    ----------
    
    min_exp : int
        The exponent in the smallest power of ten
    max_exp : int
        The exponent in the largest power of ten

    Returns
    -------

    numpy.array
        tick values
    list of str
        tick labels for each power of ten

    """

    x=np.arange(1,10)
    y=np.arange(min_exp,max_exp).astype(float)
    log_minorticks=[]
    log_majorticks=[]
    log_majorticklabels=[]
    for j in y:
        for i in x:
            log_minorticks.append(np.log10(np.round(i*10**j,int(np.abs(j)))))
            if i==1:
                log_majorticklabels.append("10$^{%d}$"%j)
                log_majorticks.append(np.log10(np.round(i*10**j,int(np.abs(j)))))

    log_minorticks=np.array(log_minorticks)
    log_majorticks=np.array(log_majorticks)
    return log_minorticks,log_majorticks,log_majorticklabels

def subplot_aerosol_dist(
    vlist,
    grid,
    cmap=cm.rainbow,
    norm=colors.Normalize(10,10000),
    xminortick_interval="1H",
    xmajortick_interval="2H",
    xticklabel_format="%H:%M",
    keep_inner_ticklabels=False,
    subplot_padding=None,
    label_subplots=False,
    label_color="white",
    column_titles=None,
    font_size=12):
    """ 
    Plot aerosol size distributions (subplots)

    Parameters
    ----------

    vlist : list of pandas.DataFrames
        Aerosol size distributions (continuous index)    
    grid : tuple (rows,columns)
        define number of rows and columns
    cmap :  matplotlib colormap
        Colormap to use, default is rainbow    
    norm : matplotlib.colors norm
        Define how to normalize the colors.
        Default is linear normalization
    xminortick_interval : str
        A pandas date frequency string 
        
        See for all options here: 
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    xmajortick_interval : str
        A pandas date frequency string
    xticklabel_format : str
        Date format string
        
        See for all options here: 
        https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-code
    keep_inner_ticklabels : bool
        If True, use ticklabels in all subplots.
        If False, use ticklabels only on outer subplots.
    subplot_padding : number or None
        Adjust space between subplots
    label_subplots : bool
        Put labels on subplots
    label_color : str
    column_titles : list of strings or None
    font_size : int
    
    Returns
    -------
    
    figure object
    array of axes objects
     
    """
    
    labels = "abcdefghijklmnopqrstuvwxyzo"
    
    assert isinstance(vlist,list)
    
    rows = grid[0]
    columns = grid[1]
    fig,ax = plt.subplots(rows,columns)
    
    if subplot_padding is not None:
        fig.tight_layout(pad=subplot_padding)
    
    ax = ax.flatten()
    
    assert len(vlist)==len(ax)
    
    ax_last = ax[-1].get_position()
    ax_first = ax[0].get_position()       
    origin = (ax_first.x0,ax_last.y0)
    size = (ax_last.x1-ax_first.x0,ax_first.y1-ax_last.y0)
    ax_width = ax_first.x1-ax_first.x0
    ax_height = ax_first.y1-ax_first.y0    
    last_row_ax = ax[-1*columns:]
    first_col_ax = ax[::columns]
    first_row_ax = ax[:columns]
    
    log_minorticks,log_majorticks,log_majorticklabels = generate_log_ticks(-10,-4)
    
    for i,vi,axi in zip(np.arange(len(ax)),vlist,ax):
        dndlogdp = vi.values.astype(float)
        tim=vi.index
        dp=vi.columns.values.astype(float)
        t1=dts.date2num(tim[0])
        t2=dts.date2num(tim[-1])
        dp1=np.log10(dp.min())
        dp2=np.log10(dp.max())
        img = axi.imshow(
            np.flipud(dndlogdp.T),
            origin="upper",
            aspect="auto",
            cmap=cmap,
            norm=norm,
            extent=(t1,t2,dp1,dp2)
        )
        axi.set_yticks(log_minorticks,minor=True)
        axi.set_yticks(log_majorticks)
        axi.set_ylim((dp1,dp2))
        
        axi.set_xlim((t1,t2))
        axi.set_xticks(pd.date_range(
            dts.num2date(t1),dts.num2date(t2),freq=xminortick_interval),minor=True)
        axi.set_xticks(pd.date_range(
            dts.num2date(t1),dts.num2date(t2),freq=xmajortick_interval))
        
        if keep_inner_ticklabels==False:
            if axi in first_col_ax:
                axi.set_yticklabels(log_majorticklabels,fontsize=font_size)
            else:
                axi.set_yticklabels([])
                
            if axi in last_row_ax:    
                axi.set_xticklabels(pd.date_range(
                    dts.num2date(t1),dts.num2date(t2),freq=xmajortick_interval).
                    strftime(xticklabel_format),fontsize=font_size)           
                for tick in axi.get_xticklabels():
                    tick.set_rotation(45)
                    tick.set_ha("right")
                    tick.set_rotation_mode("anchor")
            else:
                axi.set_xticklabels([])
        else:
            axi.set_yticklabels(log_majorticklabels,fontsize=font_size)
            axi.set_xticklabels(pd.date_range(
                dts.num2date(t1),dts.num2date(t2),freq=xmajortick_interval).
                strftime(xticklabel_format),fontsize=font_size)
            for tick in axi.get_xticklabels():
                tick.set_rotation(45)
                tick.set_ha("right")
                tick.set_rotation_mode("anchor")
        
        if label_subplots:
            axi.text(.01, .99, labels[i], ha='left', va='top', 
                color=label_color, transform=axi.transAxes, fontsize=font_size)

    if column_titles is not None:
        for column_title,axy in zip(column_titles,first_row_ax):
            axy.set_title(column_title,fontsize=font_size)
    
    if columns>1:
        xspace = (size[0]-columns*ax_width)/(columns-1.0)
    else:
        xspace = (size[1]-rows*ax_height)/(rows-1.0)
    
    c_handle = plt.axes([origin[0] + size[0] + xspace, origin[1], 0.02, size[1]])
    cbar = plt.colorbar(img,cax=c_handle)
    cbar.ax.tick_params(labelsize=font_size)
    cbar.set_label("$dN/dlogD_p$, [cm$^{-3}$]")

    return fig,ax    

def plot_aerosol_dist(
    v,
    ax,
    cmap=cm.rainbow,
    norm=colors.Normalize(10,10000),
    xminortick_interval="1H",
    xmajortick_interval="2H",
    xticklabel_format="%H:%M"):    
    """ 
    Plot aerosol particle number-size distribution surface plot

    Parameters
    ----------

    v : pandas.DataFrame or list of pandas.DataFrames
        Aerosol number size distribution (continuous index)
    ax : axes object
        axis on which to plot the data
    cmap :  matplotlib colormap
        Colormap to use, default is rainbow    
    norm : matplotlib.colors norm
        Define how to normalize the colors.
        Default is linear normalization
    xminortick_interval : str
        A pandas date frequency string
        
        See for all options here: 
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    xmajortick_interval : str
        A pandas date frequency string
    xticklabel_format : str
        Date format string
        
        See for all options here: 
        https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-code
     
    """
    handle = ax
    box = handle.get_position()
    origin = (box.x0,box.y0) 
    size = (box.width,box.height)
    handle.set_ylabel('$D_p$, [m]')
    
    tim = v.index
    dp = v.columns.values.astype(float)
    dndlogdp = v.values.astype(float)

    log_minorticks,log_majorticks,log_majorticklabels = generate_log_ticks(-10,-4)
    handle.set_yticks(log_minorticks,minor=True)
    handle.set_yticks(log_majorticks)
    handle.set_yticklabels(log_majorticklabels)
    
    t1=dts.date2num(tim[0])
    t2=dts.date2num(tim[-1])
    dp1=np.log10(dp.min())
    dp2=np.log10(dp.max())

    img = handle.imshow(
        np.flipud(dndlogdp.T),
        origin="upper",
        aspect="auto",
        cmap=cmap,
        norm=norm,
        extent=(t1,t2,dp1,dp2)
    )

    handle.set_ylim((dp1,dp2))
    handle.set_xlim((t1,t2))
    handle.set_xticks(pd.date_range(
        dts.num2date(t1),dts.num2date(t2),freq=xminortick_interval),minor=True)
    handle.set_xticks(pd.date_range(
        dts.num2date(t1),dts.num2date(t2),freq=xmajortick_interval))
    handle.set_xticklabels(pd.date_range(
        dts.num2date(t1),dts.num2date(t2),freq=xmajortick_interval).
        strftime(xticklabel_format))
        
    for tick in handle.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha("right")
        tick.set_rotation_mode("anchor")

    c_handle = plt.axes([origin[0]*1.03 + size[0]*1.03, origin[1], 0.02, size[1]])
    cbar = plt.colorbar(img,cax=c_handle)
    cbar.set_label('$dN/dlogD_p$, [cm$^{-3}$]')

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
