import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dts
from matplotlib import colors
from matplotlib.pyplot import cm
from datetime import datetime, timedelta
from scipy.optimize import minimize

def plot_one_to_one_line(ax=None, color='black', linestyle='--', linewidth=1):
    """
    Draws a one-to-one line (y = x) on the visible part of the plot.
    
    Parameters
    ----------

    ax : matplotlib.axes.Axes (optional)
        The axes to draw the line on. If `None`, uses current axes.
    color : str
        Color of the line.
    linestyle : str 
        Style of the line (e.g., `'--'` for dashed)
    linewidth : float 
        Width of the line

    """

    if ax is None:
        ax = plt.gca()
    
    # Get current axis limits
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # Define the range for the one-to-one line
    line_min = max(xmin, ymin)
    line_max = min(xmax, ymax)
    
    # Plot the one-to-one line
    ax.plot([line_min, line_max], [line_min, line_max], color=color, linestyle=linestyle, linewidth=linewidth)

    # Reset limits to ensure they don't change after plotting the line
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def set_legend_outside(ax,handles=None,labels=None,coords=(1,1),**kwargs):
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

    coords : tuple of floats or ints
        anchor point for the legend

    kwargs : 
        other parameters passed to legend  

    Returns
    -------

    Legend

    """
 
    if ((handles is not None) and (labels is not None)):
        leg = ax.legend(
            handles,
            labels,
            bbox_to_anchor=coords, 
            loc='upper left',
            borderaxespad=0,
            frameon=False,
            **kwargs)
    elif handles is not None:
        leg = ax.legend(
            handles=handles,
            bbox_to_anchor=coords, 
            loc='upper left',
            borderaxespad=0,
            frameon=False,
            **kwargs)
    elif labels is not None:
        leg = ax.legend(
            labels,
            bbox_to_anchor=coords, 
            loc='upper left',
            borderaxespad=0,
            frameon=False,
            **kwargs)
    else:
        leg = ax.legend(
            bbox_to_anchor=coords, 
            loc='upper left',
            borderaxespad=0,
            frameon=False,
            **kwargs)

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


def generate_log_ticks(min_exp,max_exp,minor=False):
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
    log_minorticklabels=[]
    for j in y:
        for i in x:
            log_minorticks.append(np.log10(np.round(i*10**j,int(np.abs(j)))))
            if i==1:
                if minor:
                    if j>-1:
                        log_majorticklabels.append(f"{10**np.log10(np.round(i*10**j,int(np.abs(j)))):.0f}")
                    else:
                        format_specifier = f'.{np.abs(np.log10(10**j)):.0f}f'
                        log_majorticklabels.append(f"{10**np.log10(np.round(i*10**j,int(np.abs(j)))):{format_specifier}}")                    
                    log_majorticks.append(np.log10(np.round(i*10**j,int(np.abs(j)))))
                else:
                    log_majorticklabels.append("10$^{%d}$"%j)
                    log_majorticks.append(np.log10(np.round(i*10**j,int(np.abs(j))))) 
            elif minor:
                if j>-1:
                    log_minorticklabels.append(f"{10**np.log10(np.round(i*10**j,int(np.abs(j)))):.0f}")
                else:
                    format_specifier = f'.{np.abs(np.log10(10**j)):.0f}f'
                    log_minorticklabels.append(f"{10**np.log10(np.round(i*10**j,int(np.abs(j)))):{format_specifier}}")
            else:
                pass

    log_minorticks=np.array(log_minorticks)
    log_minorticks=log_minorticks[log_minorticks<=max_exp]
    log_majorticks=np.array(log_majorticks)
    return log_minorticks,log_majorticks,log_majorticklabels,log_minorticklabels

def subplot_aerosol_dist(
    vlist,
    grid,
    norm="log",
    vmin=10,
    vmax=10000,
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
    norm : string
        Define how to normalize the colors.
        "linear" or "log"
    vmin : float or int
        Minimum value in colorbar
    vmax : float or int
        Maximum value in colorbar
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
    list of image handles
    colorbar handle
     
    """
     
    assert isinstance(vlist,list)

    allowed_norm_values = ["linear", "log"]

    if norm not in allowed_norm_values:
        raise ValueError(f"Invalid input: {norm}. Expected one of {allowed_norm_values}.")

    if norm=="linear":
        norm = colors.Normalize(vmin,vmax)
    if norm=="log":
        norm = colors.LogNorm(vmin,vmax)

    cmap = cm.turbo

    rows = grid[0]
    columns = grid[1]
    fig,ax = plt.subplots(rows,columns)
    
    if hspace_padding is not None:
        fig.subplots_adjust(hspace=hspace_padding)
        #fig.tight_layout(pad=subplot_padding)

    ax_row = ax.flatten() # indices go row first
    ax_col = ax.T.flatten() # indices go column first

    # image handles
    imgs = []

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
    
    log_minorticks,log_majorticks,log_majorticklabels,_ = generate_log_ticks(-10,-4)
    
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
            imgs.append(img)
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

    return fig,ax_row,imgs,cbar

def plot_aerosol_dist(
    v,
    ax,
    norm="log",
    clim=None,
    cmap="turbo",
    xmajortick_interval=None,
    xminortick_interval=None,
    xticklabel_format="%Y-%m-%d %H:%M"):    
    """ 
    Plot aerosol particle number-size distribution surface plot

    Parameters
    ----------

    v : pandas.DataFrame or list of pandas.DataFrames
        Aerosol number size distribution (continuous index)
    ax : axes object
        axis on which to plot the data
    norm : string
        Define how to normalize the colors.
        "linear" or "log"
    clim : list with two elements or None
        Minimum and maximum value in colorbar, None calculates automatic limits
    xminortick_interval : pandas date frequency string
        See for all options here: 
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    xmajortick_interval : pandas date frequency string
    xticklabel_format : str
        See for all options here: 
        https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-code
     
    Returns
    -------

    image handle
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
   
    # Find optimum interval for minor and major ticks
    
    if (xmajortick_interval is None) or (xminortick_interval is None):
        xmajortick_interval = pd.tseries.frequencies.to_offset(((tim.max() - tim.min())/10))
        xminortick_interval = pd.tseries.frequencies.to_offset((tim.max() - tim.min())/(10*2)) 
    else:
        pass

    time_minorticks,time_majorticks,time_ticklabels = generate_timeticks(
        tim[0],tim[-1],xminortick_interval,xmajortick_interval,xticklabel_format)
    handle.set_xticks(time_minorticks,minor=True)
    handle.set_xticks(time_majorticks)
    handle.set_xticklabels(time_ticklabels)
    
    log_minorticks,log_majorticks,log_majorticklabels,_ = generate_log_ticks(-10,-4)
    handle.set_yticks(log_minorticks,minor=True)
    handle.set_yticks(log_majorticks)
    handle.set_yticklabels(log_majorticklabels)
    
    t1=dts.date2num(tim[0])
    t2=dts.date2num(tim[-1])
    dp1=np.log10(dp.min())
    dp2=np.log10(dp.max())

    allowed_norm_values = ["linear", "log"]

    if norm not in allowed_norm_values:
        raise ValueError(f"Invalid input: {norm}. Expected one of {allowed_norm_values}.")

    if clim is None:
        clim = np.nanpercentile(dndlogdp,[10,90])


    if norm=="linear":
        norm = colors.Normalize(clim[0],clim[1])
    if norm=="log":
        norm = colors.LogNorm(clim[0],clim[1])

    img = handle.imshow(
        np.flipud(dndlogdp.T),
        origin="upper",
        aspect="auto",
        cmap=cm.get_cmap(cmap),
        norm=norm,
        extent=(t1,t2,dp1,dp2)
    )

    handle.set_ylim((dp1,dp2))
    handle.set_xlim((t1,t2))

    rotate_xticks(handle,45)

    c_handle = plt.axes([origin[0]*1.03 + size[0]*1.03, origin[1], 0.02, size[1]])
    cbar = plt.colorbar(img,cax=c_handle)
    cbar.set_label('$dN/dlogD_p$, [cm$^{-3}$]')

    return img, cbar
