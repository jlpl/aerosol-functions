import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dts
from matplotlib import colors
from matplotlib.pyplot import cm
from datetime import datetime, timedelta
from scipy.optimize import minimize


def set_legend_outside(ax,handles=None,labels=None,fs=10):
    """
    Put legend outside axes (upper right corner)

    Parameters
    ----------

    ax : Axes
        axes to add the legend to

    handles : list of handles
        list of lines or points in the legend

    labels : list of strings
        labels for the legend entries

    fs : int
        font size

    Returns
    -------

    Legend

    """
 
    if ((handles is not None) and (labels is not None)):
        leg = ax.legend(
            handles,
            labels,
            bbox_to_anchor=(1, 1), 
            loc='upper left',
            borderaxespad=0,
            fontsize=fs,
            frameon=False)
    elif handles is not None:
        leg = ax.legend(
            handles=handles,
            bbox_to_anchor=(1, 1), 
            loc='upper left',
            borderaxespad=0,
            fontsize=fs,
            frameon=False)
    elif labels is not None:
        leg = ax.legend(
            labels,
            bbox_to_anchor=(1, 1), 
            loc='upper left',
            borderaxespad=0,
            fontsize=fs,
            frameon=False)
    else:
        leg = ax.legend(
            bbox_to_anchor=(1, 1), 
            loc='upper left',
            borderaxespad=0,
            fontsize=fs,
            frameon=False)

    return leg


def rotate_xticks(ax,degrees):
    """
    Parameters
    ----------
    
    ax : matplotlib axes
    degrees : int or float
       number of degrees to rotate the xticklabels
    
    """
    for tick in ax.get_xticklabels():
        tick.set_rotation(degrees)
        tick.set_ha("right")
        tick.set_rotation_mode("anchor")

def show_matrix_values(
    ax,
    matrix,
    text_format="%d",
    text_color="white"):
    """
    Plot numerical values on top of the cells when
    visualizing a matrix with imshow()

    Parameters
    ----------
    
    ax : matplotlib.axes
    matrix : numpy 2d-array
    text_format : str
    text_color : str

    """
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, text_format%confmat[i, j],
                    ha='center', va='center', color=text_color)

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
    log_minorticks=log_minorticks[log_minorticks<=max_exp]
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
    hspace_padding=None,
    vspace_padding=None,
    subplot_labels=None,
    label_color="black",
    label_size=10,
    column_titles=None,
    fill_order="row",
    **kwargs):
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
    fill_order : str
        `"rows"` fills the subplots row by row
        `"columns"` fills the subplots column by column  
    **kwargs : optional parameters passed to matplotlib imshow()

    Returns
    -------
    
    figure object
    array of axes objects
    colorbar handle
     
    """
     
    assert isinstance(vlist,list)
    
    rows = grid[0]
    columns = grid[1]
    fig,ax = plt.subplots(rows,columns)
    
    if hspace_padding is not None:
        fig.subplots_adjust(hspace=hspace_padding)
        #fig.tight_layout(pad=subplot_padding)

    ax_row = ax.flatten() # indices go row first
    ax_col = ax.T.flatten() # indices go column first

    # Assert some limits regarding grid and plots
    if (rows==1) | (columns==1):
        assert len(ax_row)==len(vlist)
    else:
        assert len(vlist)<=len(ax_row)
        assert len(vlist)>columns*(rows-1)
    
    ax_last = ax_row[-1].get_position()
    ax_first = ax_row[0].get_position()
    origin = (ax_first.x0,ax_last.y0)
    size = (ax_last.x1-ax_first.x0,ax_first.y1-ax_last.y0)
    ax_width = ax_first.x1-ax_first.x0
    ax_height = ax_first.y1-ax_first.y0    
    last_row_ax = ax_row[-1*columns:]
    first_col_ax = ax_row[::columns]
    first_row_ax = ax_row[:columns]
    
    log_minorticks,log_majorticks,log_majorticklabels = generate_log_ticks(-10,-4)
    
    for i in np.arange(len(ax_row)):
        
        if (i<len(vlist)):
            vi = vlist[i]

            if fill_order=="column":
                axi = ax_col[i]
            if fill_order=="row":
                axi = ax_row[i]
            
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
                extent=(t1,t2,dp1,dp2),
                **kwargs
            )
        else:
            vi = vlist[i-columns]
            if fill_order=="column":
                axi = ax_col[i]
            if fill_order=="row":
                axi = ax_row[i]
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
            axi.axis("off")
            ax_row[i-columns].set_xticklabels(time_ticklabels)
            rotate_xticks(ax_row[i-columns],45)

    for i in np.arange(len(ax_row)):        
        if subplot_labels is not None:
            if i<len(vlist):
                if fill_order=="column":
                    axi = ax_col[i]
                if fill_order=="row":
                    axi = ax_row[i] 
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

    return fig,ax_row,cbar

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
     
    Returns
    -------

    colorbar handle
     
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

    c_handle = plt.axes([origin[0]*1.03 + size[0]*1.03, origin[1], 0.02, size[1]])
    cbar = plt.colorbar(img,cax=c_handle)
    cbar.set_label('$dN/dlogD_p$, [cm$^{-3}$]')

    return cbar
