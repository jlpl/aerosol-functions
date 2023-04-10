import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dts
from matplotlib import colors
from matplotlib.pyplot import cm
from datetime import datetime, timedelta
from scipy.optimize import minimize

def rotate_xticks(ax,degrees):
    """
    Parameters
    ----------
    
    ax : matplotlib axes
    degrees : int or float
       number of degrees to rotate the ticklabels
    
    """
    for tick in ax.get_xticklabels():
        tick.set_rotation(degrees)
        tick.set_ha("right")
        tick.set_rotation_mode("anchor")
        
def generate_timeticks(
    t_min,
    t_max,
    minortick_interval,
    majortick_interval,
    ticklabel_format):
    """
    Parameters
    ----------
    
    t_min : pandas timestamp
    t_max : pandas timestamp
    majortick_interval : pandas date frequency string
        See for all options here: 
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    minortick_interval : pandas date frequency string
    ticklabel_format : python date format string
        See for all options here: 
        https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-code
    
    Returns
    -------
    pandas DatetimeIndex
        minor tick values
    pandas DatetimeIndex
        major tick values
    pandas Index containing strings
        major tick labels

    """
    minor_ticks = pd.date_range(
        t_min,t_max,freq=minortick_interval)
    major_ticks = pd.date_range(
        t_min,t_max,freq=majortick_interval)
    major_ticklabels = pd.date_range(
        t_min,t_max,freq=majortick_interval).strftime(ticklabel_format)
        
    return minor_ticks,major_ticks,major_ticklabels


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
        minor tick values
    numpy.array
        major tick values
    list of strings
        major tick labels (powers of ten)

    """

    x=np.arange(1,10)
    y=np.arange(min_exp,max_exp+1).astype(float)
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
    subplot_labels=None,
    label_color="black",
    label_size=10,
    column_titles=None):
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
        A pandas date frequency string.
        See for all options here: 
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    xmajortick_interval : str
        A pandas date frequency string
    xticklabel_format : str
        Date format string.
        See for all options here: 
        https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-code
    keep_inner_ticklabels : bool
        If True, use ticklabels in all subplots.
        If False, use ticklabels only on outer subplots.
    subplot_padding : number or None
        Adjust space between subplots
    subplot_labels : list of str or None
        The labels to put to labels the subplots with
    label_color : str
    label_size :  float
    column_titles : list of strings or None
    
    Returns
    -------
    
    figure object
    array of axes objects
     
    """
     
    assert isinstance(vlist,list)
    
    rows = grid[0]
    columns = grid[1]
    fig,ax = plt.subplots(rows,columns)
    
    if subplot_padding is not None:
        fig.tight_layout(pad=subplot_padding)
    
    ax = ax.flatten()
    
    # Assert some limits regarding grid and plots
    if (rows==1) | (columns==1):
        assert len(ax)==len(vlist)
    else:
        assert len(vlist)<=len(ax)
        assert len(vlist)>columns*(rows-1)
    
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
    
    for i in np.arange(len(ax)):
        
        if (i<len(vlist)):
            vi = vlist[i]
            axi = ax[i]
            
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
        else:
            vi = vlist[i-columns]
            axi=ax[i]
            tim=vi.index
        
        time_minorticks,time_majorticks,time_ticklabels = generate_timeticks(
            tim[0],tim[-1],xminortick_interval,xmajortick_interval,xticklabel_format)
        
        axi.set_yticks(log_minorticks,minor=True)
        axi.set_yticks(log_majorticks)
        axi.set_ylim((dp1,dp2))
        
        axi.set_xticks(time_minorticks,minor=True)
        axi.set_xticks(time_majorticks)
        axi.set_xlim((t1,t2))
        
        if keep_inner_ticklabels==False:
            if axi in first_col_ax:
                axi.set_yticklabels(log_majorticklabels)
            else:
                axi.set_yticklabels([])
                
            if axi in last_row_ax:
                axi.set_xticklabels(time_ticklabels)
                rotate_xticks(axi,45)
            else:
                axi.set_xticklabels([])
        else:
            axi.set_yticklabels(log_majorticklabels)
            axi.set_xticklabels(time_ticklabels)
            rotate_xticks(axi,45)
            
        if i>=len(vlist):
            axi.spines[['right','top','left','bottom']].set_visible(False)
            axi.set_yticks([],minor=True)
            axi.set_yticks([])
            axi.set_yticklabels([])
        
        if subplot_labels is not None:
            if i<len(vlist):
                axi.text(.01, .99, subplot_labels[i], ha='left', va='top', 
                    color=label_color, transform=axi.transAxes, fontsize=label_size)

    if column_titles is not None:
        for column_title,axy in zip(column_titles,first_row_ax):
            axy.set_title(column_title)
    
    if columns>1:
        xspace = (size[0]-columns*ax_width)/(columns-1.0)
    else:
        xspace = (size[1]-rows*ax_height)/(rows-1.0)
    
    c_handle = plt.axes([origin[0] + size[0] + xspace, origin[1], 0.02, size[1]])
    cbar = plt.colorbar(img,cax=c_handle)

    return fig,ax,cbar

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
    xminortick_interval : pandas date frequency string
        See for all options here: 
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    xmajortick_interval : pandas date frequency string
    xticklabel_format : str
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
    
    time_minorticks,time_majorticks,time_ticklabels = generate_timeticks(
        tim[0],tim[-1],xminortick_interval,xmajortick_interval,xticklabel_format)
    handle.set_xticks(time_minorticks,minor=True)
    handle.set_xticks(time_majorticks)
    handle.set_xticklabels(time_ticklabels)
    
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
        
    rotate_xticks(handle,45)

    c_handle = plt.axes([origin[0]*1.03 + size[0]*1.03, origin[1], 0.02, size[1]])
    cbar = plt.colorbar(img,cax=c_handle)
    cbar.set_label('$dN/dlogD_p$, [cm$^{-3}$]')
    


def stacked_plots(df,height_coef=3,spacing_coef=1.5,plot_type="plot",color=None,cmap=None,**kwargs):
    """
    Vertically stacked overlapping plots
    
    Parameters
    ----------
    df : pandas.DataFrame
        first column is plotted to lower/foremost plot
    height_coef : int or float
        scales the height of the subplots
    spacing_coef : int or float
        scales the vertical spacing between subplots
    plot_type : str
        "scatter", "plot" or "fill_between" corresponding to
        the matplotlib functions of the same name.
    color : matplotlib color
    cmap : matplotlib colormap
        can apply to any plot_type
    **kwargs : optional properties passed on to the plot_type
        
    Returns
    -------
    matplotlib figure
    matplotlib axes
        lowest subplot x-axes
    matplotlib axes
        lowest subplot y-axes
    """
    fig = plt.figure()
    n = df.shape[1]
    
    h = 1.0/n
    ax = []
    
    ylabels = df.columns
    y = df.values
    x = df.index.values
    
    # lowest plot is the foremost
    n_idx = np.flip(np.arange(n))

    for i in n_idx:
        rect = [0,i*h*spacing_coef,1,h*height_coef]
        ax.append(fig.add_axes(rect))
    
    ax = np.flip(np.array(ax))
    ax_idx = np.arange(len(ax))
    
    for i,axi in zip(ax_idx,ax):
        if i==0:
            axi.spines[['top','left','right']].set_visible(False)
            axi.set_yticks([])
            axi.set_yticklabels([])
            axy = axi.twinx()
            axy.spines[['top','left','bottom']].set_visible(False)
            axy.set_xlim((x.min(),x.max()))
            axy.set_ylim((y.min(),y.max()))
        elif (i==ax_idx[-1]):
            axi.spines[['right','top','left','bottom']].set_visible(False)
            axi.set_yticks([])
            axi.set_yticklabels([])
            axi.set_xticks([])
            axi.set_xticklabels([])
        else:
            axi.spines[['right','top','left','bottom']].set_visible(False)
            axi.set_yticks([])
            axi.set_yticklabels([])
            axi.set_xticks([])
            axi.set_xticklabels([])
        
        axi.set_ylabel(ylabels[i], rotation=0, y=0, rotation_mode="anchor",
            verticalalignment='bottom',horizontalalignment="right")
        
        axi.patch.set_alpha(0.0)
        
        if plot_type == "plot":
            if color is not None:
                axi.plot(x,y[:,i],c=color, **kwargs)
            elif cmap is not None:
                axi.plot(x,y[:,i],c=cmap(float(i)/float(n)), **kwargs)
            else:
                axi.plot(x,y[:,i], **kwargs)
                
        if plot_type == "scatter":
            if color is not None:
                axi.scatter(x,y[:,i],c=color,**kwargs)
            elif cmap is not None:
                axi.scatter(x,y[:,i],c=y[:,i], cmap=cmap, norm=colors.Normalize(y.min(),y.max()),**kwargs)
            else:
                axi.scatter(x,y[:,i],**kwargs)
                
        if plot_type=="fill_between":
            if color is not None:
                axi.fill_between(x,y[:,i],color=color, **kwargs)
            elif cmap is not None:
                axi.fill_between(x,y[:,i],color=cmap(float(i)/float(n)), **kwargs)
            else:
                axi.fill_between(x,y[:,i], **kwargs)
        
        axi.set_xlim((x.min(),x.max()))
        axi.set_ylim((y.min(),y.max()))
    
    return fig,ax[0],axy
