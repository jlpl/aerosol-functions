from functools import partial
from bokeh.io import curdoc
from bokeh.layouts import column, row, Spacer
from bokeh.models import TapTool, FreehandDrawTool, BoxEditTool, ColumnDataSource, Button, Div, TextInput, FileInput, CustomJS, Span, Scatter, TextAreaInput, Toolbar, Select
from bokeh.plotting import figure
from bokeh.events import ButtonClick
import numpy as np
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from bokeh.transform import linear_cmap
from bokeh.models import ColorBar, LogColorMapper
from bokeh import palettes
from bokeh.palettes import Turbo256 as palette
from bokeh.palettes import Category10
import pandas as pd
import json
import sys
from io import StringIO
import base64
import aerosol.functions as af
import aerosol.fitting as afi
from datetime import datetime, timezone
from matplotlib.path import Path
from sklearn.mixture import GaussianMixture

from kneed import KneeLocator
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline


### TODO: 
# Add box select tool
# Add dropown menu for the colormap 
#
# Button 1: Fit all
# Button 2: Fit selected
# Button 3: Clear all
# Button 4: Clear Selected
# Button 5: Check dist should be on all the time


def find_df_in_polygon(data,poly):
    # Return aerosol number size disribution from inside a rectangle 
    # that encloses the selected polygon. 
    # Polygon values not inside the polygon are NaN
    #
    # index: milliseconds since epoch
    # columns:  diameters  in meters
    # data: dN/dlogDp
    
    x_coords = poly['x'] # milliseconds since epoch
    y_coords = poly['y'] # diameter

    # Select part of dataframe that contains all of the polygon
    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    y_min = np.min(y_coords)
    y_max = np.max(y_coords)
    
    data.index = (data.index - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")

    index_array = data.index.values
    column_array = data.columns.astype(float).values

    data_subset = data.iloc[((index_array >= x_min) & (index_array <= x_max)), 
        ((column_array >= y_min) & (column_array <= y_max))]

    loop = Path(np.column_stack((x_coords,y_coords)).astype(float))

    flattened = data_subset.stack().reset_index()
    flattened.columns = ['x','y','value']
    coordinates = flattened[['x','y']].values.astype(float)

    inside = loop.contains_points(coordinates)

    points_inside = flattened[inside]

    mask = pd.DataFrame(False, index=data_subset.index, columns=data_subset.columns)
    
    for idx, col in points_inside[['x','y']].to_numpy():
        mask.loc[idx, col] = True
    
    data_subset = data_subset.where(mask, np.nan)

    return data_subset

def fit_modes_method(data):
    global fit_modes_text_input
    global info_div

    num_modes = int(fit_modes_text_input.value)
    tim = data.index.values.astype(float)
    dp = np.log10(data.columns.astype(float).values*1e9)

    peaks = []

    info_text_updates = ""

    for j in range(data.shape[0]):
        conc = data.iloc[j,:].values.flatten()

        res = afi.fit_multimode(dp, conc, "1970-01-01 00:00", n_modes = num_modes)

        for r in res["gaussians"]: 
            peak = {
                "time": tim[j], # milliseconds since epoch
                "diam": (10**r["mean"])*1e-9, # meters
                "mean": r["mean"],
                "sigma": r["sigma"], 
                "amplitude": r["amplitude"]
            }

            peaks.append(peak)
        
        time_str = pd.to_datetime(tim[j],unit="ms").strftime("%Y-%m-%d %H:%M:%S")
        info_text_updates += f"Fitted: {time_str} <br>"

        info_div.text = info_text_updates

    return peaks

#def fit_maxconc_method(data):
#    global info_div
#
#    tim = data.index.values.astype(float)
#    dp = np.log10(data.columns.astype(float).values*1e9)
#
#    peaks = []
#
#    info_text_updates = ""
#
#    for j in range(data.shape[1]):
#        conc = data.iloc[:,j].values.flatten()
#
#        # Convert to pandas Series
#        ds = pd.Series(index = tim, data = conc)
#    
#        # Interpolate away the NaN values but do not extrapolate, remove any NaN tails
#        s = ds.interpolate(limit_area="inside").dropna()
#    
#        # Set negative values to zero
#        s[s<0]=0
#    
#        # Recover x and y for fitting
#        x_interp = s.index.values
#        y_interp = s.values
#
#        # Ensure there is enough data    
#        if ((len(x_interp)<3) & (len(y_interp)<3)):
#            continue
#
#        # Perform fit
#        coef = np.trapz(y_interp,x_interp)
#
#        samples = af.sample_from_dist(x_interp,y_interp,10000)
#
#        res = afi.fit_gmm(samples, 1, coef, None, None)
#
#        peak = {
#            "time": res[0]["mean"], # milliseconds since epoch
#            "diam": (10**dp[j])*1e-9, # meters
#            "mean": res[0]["mean"],
#            "sigma": res[0]["sigma"], 
#            "amplitude": res[0]["amplitude"]
#        }
#
#        peaks.append(peak)
#
#        dp_str = f"{10**dp[j]:.3f}"
#        info_text_updates += f"Fitted: {dp_str} nm <br>"
#
#        info_div.text = info_text_updates
#
#    return peaks

def update_peaks():
    global list_of_polygons
    global selected_polygon_index
    global mode_peaks_source
    #global maxconc_peaks_source

    peak_mode_diams = []
    peak_mode_times = []

#    peak_maxconc_diams = []
#    peak_maxconc_times = []

    for p in list_of_polygons:
        peak_mode_diams = peak_mode_diams + [params["diam"] for params in p["fit_mode_params"]]
        peak_mode_times = peak_mode_times + [params["time"] for params in p["fit_mode_params"]]
#        peak_maxconc_diams = peak_maxconc_diams + [params["diam"] for params in p["fit_maxconc_params"]]
#        peak_maxconc_times = peak_maxconc_times + [params["time"] for params in p["fit_maxconc_params"]]

    mode_peaks_source.data = {
        "t": peak_mode_times,
        "d": peak_mode_diams
    }
#    maxconc_peaks_source.data = {
#        "t": peak_maxconc_times, 
#        "d": peak_maxconc_diams
#    }

def do_fit_modes(event):
    # Run this when the fit modes button is pressed
    global list_of_polygons
    global selected_polygon_index
    global df

    if selected_polygon_index is not None:

        polygon = list_of_polygons[selected_polygon_index]
        
        df_subset = find_df_in_polygon(df.copy(),polygon)
        
        peaks = fit_modes_method(df_subset)

        list_of_polygons[selected_polygon_index]["fit_mode_params"] = peaks

        update_peaks()

def do_fit_mode(event):
    # Run this when the fit modes button is pressed
    global list_of_polygons
    global selected_polygon_index
    global df
    global dist

    if selected_polygon_index is not None:
        df_subset = dist.copy()

        polygon = list_of_polygons[selected_polygon_index]

        df_subset.index =  (df_subset.index - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")

        # Do the fit to the dist        
        peaks = fit_modes_method(df_subset)

        # Update inside the selected polygon only the selected dist
        for p in list_of_polygons[selected_polygon_index]["fit_mode_params"]:
            if (p["time"]==peaks["time"]):
                p = peaks

        update_peaks()

#def do_fit_maxconc(event):
#    # Run this when the fit maxconc button is pressed
#    global list_of_polygons
#    global selected_polygon_index
#    global df
#
#    if selected_polygon_index is not None:
#
#        polygon = list_of_polygons[selected_polygon_index]
#        
#        df_subset = find_df_in_polygon(df.copy(),polygon)
#        
#        peaks = fit_maxconc_method(df_subset)
#
#        list_of_polygons[selected_polygon_index]["fit_maxconc_params"] = peaks
#
#        update_peaks()


def remove_mode_fits_from_polygon(event):
    # a remove mode fits button is pressed
    global list_of_polygons
    global selected_polygon_index

    if selected_polygon_index is not None:

        # Empty the list containing the fitted parametrs
        list_of_polygons[selected_polygon_index]["fit_mode_params"] = []

        # Remove the points from the plot
        update_peaks()


def remove_mode_fit_from_polygon(event):
    # a remove mode fits button is pressed
    global list_of_polygons
    global selected_polygon_index
    global dist

    if selected_polygon_index is not None:
        df_subset = dist.copy()

        dist_time =  (dist.index[0] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")
       
        for i in range(len(list_of_polygons[selected_polygon_index]["fit_mode_params"])):
            if (list_of_polygons[selected_polygon_index]["fit_mode_params"][i]["time"]==dist_time):
                list_of_polygons[selected_polygon_index]["fit_mode_params"][i] = []
                
        # Remove the point from the plot
        update_peaks()

#def remove_maxconc_fits_from_polygon(event):
#    # a remove mode fits button is pressed
#    global list_of_polygons
#    global selected_polygon_index
#
#    if selected_polygon_index is not None:
#
#        # Empty the list containing the fitted parametrs
#        list_of_polygons[selected_polygon_index]["fit_maxconc_params"] = []
#
#        # Remove the points from the plot
#        update_peaks()


def update_selected_polygon(attr, old, new):
    # When you click on a polygon using the Tap tool
    global list_of_polygons
    global selected_polygon_index
    global label_submit_button
    global fig
    global polygon_background_source
    global info_div


    # A polygon has been selected using the Tap tool
    if (len(new)==1):
        selection_id = new[0]
        selected_x = polygon_background_source.data["xs"][selection_id]
        selected_y = polygon_background_source.data["ys"][selection_id]
        
        # Find the polygon based on the x coordinates
        for i in range(len(list_of_polygons)):
            if (list_of_polygons[i]['x'] == selected_x):
                selected_polygon_index=i

                info_div.text = f""" \
                Selected ROI: <br> \
                Index: {selected_polygon_index} <br> \
                Label: {list_of_polygons[selected_polygon_index]["label"]} \
                """
    else:
        selected_polygon_index = None
        info_div.text = f""


def update_polygon_renderer():
    global list_of_polygons
    global polygon_background_source
    
    loaded_data = {"xs": [polygon['x'] for polygon in list_of_polygons],
                   "ys": [polygon['y'] for polygon in list_of_polygons]} 
 
    polygon_background_source.data = loaded_data


def update_list_of_polygons(attr, old, new):
    # When you create or remove polygons
    global list_of_polygons
    global polygon_source

    new_x = new['xs']
    old_x = old['xs']
    new_y = new['ys']
    old_y = old['ys']

    # A polygon was added
    if len(new_x)>0:
    # if (len(new_x)>len(old_x)):
        #added_x = find_difference(new_x,old_x) 
        #added_y = find_difference(new_y,old_y)

        polygon={}
        #polygon["x"] = added_x
        #polygon["y"] = added_y
        polygon["x"] = new_x[0]
        polygon["y"] = new_y[0]
        polygon["fit_mode_params"] = []
        polygon["fit_maxconc_params"] = []
        polygon["label"] = ""
    
        list_of_polygons.append(polygon)
        update_polygon_renderer()

        polygon_source.data = dict(xs=[], ys=[])

    else:
        return


def load_aerosol_file(event):
    global fig
    global data_source
    global mask_source
    global dist_source
    global polygon_background_source
    global mapper
    global info_div
    global df
    global mode_peaks_source
    #global maxconc_peaks_source

    f = aerosol_file_text_input.value 

    try:
        df = pd.read_csv(f, index_col=0, parse_dates=True)

        x = df.index.values
        y = df.columns.values.astype(float)
        z = df.values.astype(float).T

        # Update the image
        data_source.data = dict(
            img=[z], 
            x=[np.min(x)], 
            y=[np.min(y)], 
            dw=[np.max(x)-np.min(x)], 
            dh=[np.max(y)-np.min(y)])

        z_mask = np.empty(z.shape, dtype=np.uint32)
        view = z_mask.view(dtype=np.uint8).reshape((z.shape[0], z.shape[1], 4))
        view.fill(0)
        
        mask_source.data = dict(
            img=[z_mask], 
            x=[np.min(x)], 
            y=[np.min(y)], 
            dw=[np.max(x)-np.min(x)], 
            dh=[np.max(y)-np.min(y)])

        fig.x_range.start = np.min(x)
        fig.x_range.end = np.max(x)
        fig.y_range.start = np.min(y)
        fig.y_range.end = np.max(y)

        # Clear the polygon data
        polygon_background_source.data = {"xs":[],"ys":[]} 

        # Add the maxconc ploints
        mode_peaks_source.data = {"t":[], "d":[]}
        #maxconc_peaks_source.data = {"t":[], "d":[]}

        info_div.text = f"Loaded aerosol data from: {f}"
    except:
        info_div.text = f"Unable to load aerosol data from: {f}"

def update_clim(event):
    global mapper
    global clim_max_text_input
    global clim_min_text_input
    global fig

    try:
        min_value=float(clim_min_text_input.value)
        max_value=float(clim_max_text_input.value)

        mapper.update(low=min_value,high=max_value)
        curdoc().add_next_tick_callback(partial(update_clim, fig))

    except:
        return

def load_polygon_data(event):
    global list_of_polygons
    global polygon_source
    global json_load_text_input
    global info_div

    # Load the json file
    with open(json_load_text_input.value, 'r') as f:
        try: 
            list_of_polygons = json.load(f)
    
            # Extract the coordinates of the ROIs
            loaded_data = {"xs": [polygon['x'] for polygon in list_of_polygons],
                           "ys": [polygon['y'] for polygon in list_of_polygons]} 
        
            polygon_background_source.data = loaded_data

            update_polygon_renderer()

            update_peaks()

            info_div.text=f"ROIs loaded from: {json_load_text_input.value}"
        except: 
            info_div.text=f"Unable to load ROIs from: {json_load_text_input.value}"

def save_polygon_data(event):
    global list_of_polygons
    global json_save_text_input
    global info_div
    
    with open(json_save_text_input.value, "w") as f:
        try:
            json.dump(list_of_polygons, f, indent=4)
            info_div.text=f"ROIs saved to: {json_save_text_input.value}"
        except:
            info_div.text=f"Unable to save ROIs to: {json_save_text_input.value}"

def tap_callback(event):
    global fig_dist
    global mask_source
    global dist_source
    global df
    global dist
    global is_check_dist
    global list_of_polygons
    global dist_fit_line_renderers
    global fit_line_colors

#    if is_check_dist:
    if df is not None:

        target_datetime = pd.to_datetime(event.x, unit='ms')

        closest_datetime = df.index[df.index<=target_datetime].max()

        x_data = df.index.values
        y_data = df.columns.values.astype(float)
        z_data = df.values.astype(float).T

        closest_index = np.argwhere(df.index==closest_datetime).flatten()

        z_mask = mask_source.data["img"][0]
        z_mask.fill(0)
        view = z_mask.view(dtype=np.uint8).reshape((z_mask.shape[0], z_mask.shape[1], 4))
        view[:,:,:3] = 255
        view[:,:, 3] = 0
        view[:,closest_index, 3] = 80 
         
        mask_source.data["img"] = [z_mask]

        dist = df.loc[closest_datetime,:]
        fig_dist.title.text=f'{closest_datetime}'
        dist_source.data = dict(x=y_data, y=dist.values)

        # remove existing lines if they exist
        for dist_fit_line_renderer in dist_fit_line_renderers:
            fig_dist.renderers.remove(dist_fit_line_renderer)

        dist_fit_line_renderers = []

        color_index = 0

        # Plot any fits if they exist
        for p in list_of_polygons:
            for d in p["fit_mode_params"]:
                if (pd.to_datetime(d["time"], unit="ms") == closest_datetime):
                    dist_fit = afi.gaussian(np.log10(y_data*1e9), d["amplitude"], d["mean"], d["sigma"])
                    renderer = fig_dist.line(
                        'x',
                        'y',
                        source=ColumnDataSource(data=dict(x=y_data,y=dist_fit)),
                        line_width=1,
                        color=fit_line_colors[color_index % len(fit_line_colors)]
                    )
                    dist_fit_line_renderers.append(renderer)
                    color_index += 1


#def toggle_check_dist(event):
#    global is_check_dist
#    global vline
#    global fig_dist
#    global mask_source
#    global check_dist_button
#    global dist_source
#    global dist_fit_source
#
#    if is_check_dist:
#        z_mask = mask_source.data["img"][0]
#        z_mask.fill(0)
#        mask_source.data["img"] = [z_mask]
#        
#        dist_source.data = dict(x=[], y=[])
#        #dist_fit_source.data = dict(x=[], y=[])
#
#        fig_dist.title.text=f' '
#        is_check_dist = False
#        check_dist_button.button_type="danger"
#    else:
#        is_check_dist = True
#        check_dist_button.button_type="success"


def add_label(event):
    global list_of_polygons
    global selected_polygon_index
    global info_div
    global dropdown

    # if we have a polygon selected
    if selected_polygon_index is not None:
        list_of_polygons[selected_polygon_index]["label"] = dropdown.value

        info_div.text = f""" \
        Selected ROI: <br> \
        Index: {selected_polygon_index} <br> \
        Label: {list_of_polygons[selected_polygon_index]["label"]} \
        """


def remove_selected_polygon(event):
    global list_of_polygons
    global selected_polygon_index
    global polygon_background_source

    if selected_polygon_index is not None:
        del list_of_polygons[selected_polygon_index]
        update_polygon_renderer()
        update_peaks()
        selected_polygon_index = None
    
def add_option(event):
    global add_option_text_input
    global dropdown

    if add_option_text_input.value not in dropdown.options:
        dropdown.options.append(add_option_text_input.value)
        add_option_text_input.value = ""

def remove_option(event):
    global add_option_text_input
    global dropdown

    if ((add_option_text_input.value in dropdown.options) & (add_option_text_input.value!="")): 
        dropdown.options.remove(add_option_text_input.value)
        add_option_text_input.value = ""

#def on_cmap_select(event):
#    #TODO Update to new colormap that was selected


def close_app():
    sys.exit()


# CLOSE THE APP
close_js = CustomJS(code="window.close()")

# Initialize regular global variables
df = None # Holds the full timeseries
dist = None # Holds the selected distribution

list_of_polygons = []
selected_polygon_index = None

#is_check_dist = False # Determines whether to plot the selected dist

# Some dummy data
zero_image = np.zeros((10, 10))
zero_image_mask = np.empty(zero_image.shape, dtype=np.uint32)
view = zero_image_mask.view(dtype=np.uint8).reshape((zero_image.shape[0], zero_image.shape[1], 4))
view.fill(0)

# Plot data sources
data_source = ColumnDataSource(data={'img': [zero_image], 'x':[pd.to_datetime("1970-01-01")], 'y':[0], 'dw':[pd.Timedelta(days=1)], 'dh':[1]})
mask_source = ColumnDataSource(data={'img': [zero_image_mask], 'x':[pd.to_datetime("1970-01-01")], 'y':[0], 'dw':[pd.Timedelta(days=1)], 'dh':[1]})
dist_source = ColumnDataSource(data=dict(x=[], y=[]))

fit_line_colors = Category10[10]
dist_fit_line_renderers = []

polygon_source = ColumnDataSource(data=dict(xs=[], ys=[]))

polygon_background_source = ColumnDataSource(data=dict(xs=[], ys=[]))

mode_peaks_source = ColumnDataSource(data = dict(t=[], d=[]))
#maxconc_peaks_source = ColumnDataSource(data = dict(t=[], d=[]))

# Create the surface plot figures
fig = figure(
    width=1000,
    height=350,
    y_axis_type="log",
    x_axis_type='datetime',
    x_axis_label="Time",
    y_axis_label="Dp, [m]")

mapper = LogColorMapper(palette=palette, low=10, high=10000)

# Create the data image
fig.image(image="img",
    source=data_source, 
    x="x", 
    y="y", 
    dw="dw", 
    dh="dh",
    color_mapper=mapper)

# Create the mask
fig.image_rgba(image="img",
    source=mask_source,
    x='x',
    y='y',
    dw='dw',
    dh='dh'
)

fig_dist = figure(
    width=500,
    height=300,
    y_axis_type="linear",
    x_axis_type="log",
    x_axis_label="Dp, [m]",
    y_axis_label="dN/dlogDp, [cm-3]",
    title = " "
)

fig_dist.line('x', 'y', source=dist_source, line_width=2, color="navy")

#fig_dist.line('x', 'y', source=dist_fit_source, legend_field="line_label", line_width=1, color="red")

poly_renderer = fig.patches('xs', 'ys', source=polygon_source, 
    fill_color="black", 
    fill_alpha=0.2,
    line_width=1,
    line_color="black")

poly_background_renderer = fig.patches('xs', 'ys', source=polygon_background_source, 
    fill_color="black", 
    fill_alpha=0.2,
    line_width=1,
    line_color="black")

# Fitted mode peaks
mode_peaks_glyph = Scatter(x="t", y="d", size=15, marker="dot", fill_color="black", line_color="black")
fig.add_glyph(mode_peaks_source, mode_peaks_glyph)

# Fitted maxconc peaks
#maxconc_peaks_glyph = Scatter(x="t", y="d", size=15, marker="dot", fill_color="black", line_color="black")
#fig.add_glyph(maxconc_peaks_source, maxconc_peaks_glyph)

# Draw tool for drawing the polygons
draw_tool = FreehandDrawTool(renderers=[poly_renderer])
fig.add_tools(draw_tool)

## Add Tap tool
tap_tool = TapTool(renderers=[poly_background_renderer])
fig.add_tools(tap_tool)

# Add colorbar
color_bar = ColorBar(
    color_mapper=mapper, 
    label_standoff=12, 
    location=(0,0), 
    title="dN/dlogDp [cm-3]")

fig.add_layout(color_bar, 'right')

# When polygon is selected with TapTool
poly_background_renderer.data_source.selected.on_change('indices', update_selected_polygon)

# Actions for tapping figure
fig.on_event('tap',tap_callback)

# When we draw we want to launch the function to update polygons
polygon_source.on_change('data', update_list_of_polygons)

# Fit mode controls
fit_modes_text_input = TextInput(placeholder = "Number of Modes", value="1", width=150, height=30)

fit_modes_button = Button(label="Fit modes", button_type="primary", width=150, height=30)

remove_modes_button = Button(label="Clear mode fit", button_type="primary", width=150, height=30)

fit_modes_button.on_event(ButtonClick, do_fit_modes)

remove_modes_button.on_event(ButtonClick, remove_mode_fits_from_polygon)

# Fit maxconc controls
#fit_maxconc_button = Button(label="Fit maxconc", button_type="primary", width=150, height=30)
#
#remove_maxconc_button = Button(label="Clear maxconc fit", button_type="primary", width=150, height=30)
#
#fit_maxconc_button.on_event(ButtonClick, do_fit_maxconc)
#
#remove_maxconc_button.on_event(ButtonClick, remove_maxconc_fits_from_polygon)
#
#button_spacer = Spacer(width=160,height=30)

# Load the aerosol data
aerosol_file_button = Button(label="Load aerosol data", button_type="primary",height=30,width=150)

aerosol_file_text_input = TextInput(placeholder="Aerosol distribution filepath",height=30,width=310)
       
aerosol_file_button.on_event(ButtonClick, load_aerosol_file)

# Update color limits
clim_min_text_input = TextInput(placeholder="Min. value:",value="10",width=150,height=30)

clim_max_text_input = TextInput(placeholder="Max. value:",value="10000",width=150,height=30)

clim_update_button = Button(label="Update color limits", button_type="primary",width=150,height=30)
    
clim_update_button.on_event(ButtonClick, update_clim)

# Load ROIs
json_load_button = Button(label="Load ROIs", button_type="primary",height=30,width=150)

json_load_text_input = TextInput(placeholder="ROI filepath",height=30,width=310)

json_load_button.on_event(ButtonClick, load_polygon_data)

# Load ROIs
#roi_file_input = FileInput(title='Load ROIs')

#roi_file_input.on_change('value', load_polygon_data)

# Save ROIs
json_save_button = Button(label="Save ROIs", button_type="primary",height=30,width=150)

json_save_text_input = TextInput(placeholder="ROI filepath",height=30,width=310)

json_save_button.on_event(ButtonClick, save_polygon_data)

# Check individual distribution button
#check_dist_button = Button(label="Show/Hide Distribution", button_type="danger",height=30,width=150)

#check_dist_button.on_event(ButtonClick, toggle_check_dist)

## Label ROIs 
#label_text_input = TextInput(placeholder="ROI label", width=310, height=30)
#
#label_submit_button = Button(label="Submit Label",button_type="primary",height=30,width=150)
#
#label_submit_button.on_event(ButtonClick, add_label)


# colormap dropdown
cmap_dropdown = Select(value = "Turbo", options = ["Turbo","Plasma","Rainbow"], width=230, height=30)

# Implement onSelect()

# Label select
dropdown = Select(options = [], width=230, height=30)

label_submit_button = Button(label="Submit Label",button_type="primary",height=30,width=150)

label_submit_button.on_event(ButtonClick, add_label)

add_option_text_input = TextInput(placeholder="Label", width=230, height=30)

add_option_button = Button(label="Add Label", button_type="primary", height=30, width=150)

add_option_button.on_event(ButtonClick, add_option)

remove_option_button = Button(label="Remove Label", button_type="primary", height=30, width=150)

remove_option_button.on_event(ButtonClick, remove_option)




# Delete ROI button
remove_selected_polygon_button = Button(label="Remove selected ROI", button_type="warning", height=30, width=150)

remove_selected_polygon_button.on_event(ButtonClick, remove_selected_polygon)

# close the app button
close_app_button = Button(label="Close App", button_type="danger")
close_app_button.js_on_click(close_js)
close_app_button.on_click(close_app)

info_div = Div(text="", width=470, height=120, styles={
    "overflow-y": "scroll", 
    "overflow-x": "scroll",
    "border":"1px solid gray",
    "border-radius": "5px",
    "padding": "10px"})

title_div = Div(text="Aerosol Size Distribution GUI Tool", width=500, height=80, styles = {
        "font-size": "30px",
        "font-weight": "bold",
        "display": "flex",
        "align-items": "flex-end"
})

# Make a layout
layout = column(
    row(title_div),
    column(
        row(clim_min_text_input,clim_max_text_input,clim_update_button),
        row(cmap_dropdown),
        row(aerosol_file_text_input,aerosol_file_button),
        row(json_load_text_input,json_load_button),
        row(json_save_text_input,json_save_button),
        width=500),
    fig,
    row(
        column(
            #row(check_dist_button), 
            row(remove_selected_polygon_button),
            row(fit_modes_text_input,fit_modes_button,remove_modes_button),
            #row(button_spacer,fit_maxconc_button,remove_maxconc_button),
            row(dropdown, add_option_text_input), 
            row(label_submit_button,add_option_button, remove_option_button),
            row(info_div),
            row(close_app_button),
            width=500),
        fig_dist
    ),
    width=1000,
)

# Add the layout to the current document
curdoc().add_root(layout)
