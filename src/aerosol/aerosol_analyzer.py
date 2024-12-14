from functools import partial
from bokeh.io import curdoc
from bokeh.layouts import column, row, Spacer
from bokeh.models import TapTool, FreehandDrawTool, ColumnDataSource, Button, Div, TextInput, FileInput, CustomJS, Span
from bokeh.plotting import figure
from bokeh.events import ButtonClick
import numpy as np
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from bokeh.transform import linear_cmap
from bokeh.models import ColorBar, LogColorMapper
from bokeh.palettes import Turbo256 as palette
import pandas as pd
import json
import sys
from io import StringIO
import base64

def update_selected_polygon(attr, old, new):
    # When you click on a polygon using the Tap tool
    global list_of_polygons
    global selected_polygon_index

    if (len(new)==1):
        selection_id = new[0]
        selected_x = polygon_source.data["xs"][selection_id]
        selected_y = polygon_source.data["ys"][selection_id]
        
        # Find the polygon based on the x coordinates
        for i in range(len(list_of_polygons)):
            if (list_of_polygons[i]['x'] == selected_x):
                selected_polygon_index=i
                
                begin_timestamp = np.min(selected_x)

                begin_dt = pd.to_datetime(begin_timestamp, unit='ms')
                begin_time = begin_dt.strftime('%Y-%m-%d %H:%M:%S')                

                end_timestamp = np.max(selected_x)
                end_dt = pd.to_datetime(end_timestamp, unit='ms')
                end_time = end_dt.strftime('%Y-%m-%d %H:%M:%S')            

                # Print out some polygon info
                info_div.text = f"ROI info:<br>      \
                Index = {selected_polygon_index}<br> \
                Begin time = {begin_time}<br>        \
                End time = {end_time}<br>            \
                Growth rate = <br>                   \
                Formation rate = <br>                      \
                "

    else:
        selected_polygon_index = None

def update_list_of_polygons(attr, old, new):
    # When you create or remove polygons
    global list_of_polygons

    new_x = new['xs']
    old_x = old['xs']

    new_y = new['ys']
    old_y = old['ys']

    # polygon added
    if (len(new_x)>len(old_x)):
        added_x = find_difference(new_x,old_x) 
        added_y = find_difference(new_y,old_y)

        polygon={}
        polygon["x"] = added_x
        polygon["y"] = added_y

        list_of_polygons.append(polygon)

    # polygon removed
    if (len(new_x)<len(old_x)):
        removed_x = find_difference(new_x,old_x) 
        removed_y = find_difference(new_y,old_y)
        
        #find the polygon that was removed
        new_list_of_polygons = [d for d in list_of_polygons if d['x'] != removed_x]
        list_of_polygons = new_list_of_polygons
        
def find_difference(list1, list2):
    # Helper function
    difference = [lst for lst in list1 if lst not in list2]

    if not difference:
        difference = [lst for lst in list2 if lst not in list1]

    if len(difference)>0:
        return difference[0]
    else:
        return []

def fit_modes(event):
    global fit_modes_text_input
    global info_div

    number_of_modes = fit_modes_text_input.value     
    if (number_of_modes.isdigit()):
        if (int(new)>0):
            pass
        else:
            info_div.text = f"Cannot fit modes if n={number_of_modes} "
            return
    else:
        info_div.text = f"Cannot fit modes if n={number_of_modes} "
        return

    return

def load_aerosol_file(event):
    global fig
    global data_source
    global polygon_source
    global poly_renderer
    global mapper
    global info_div
    global df

    f=aerosol_file_text_input.value 

    try:
        df = pd.read_csv(f, index_col=0, parse_dates=True)

        x = df.index.values
        y = df.columns.values.astype(float)
        z = df.values.astype(float).T
       
        # update the image
        data_source.data["img"] = [z]
       
        fig.x_range.start = np.min(x)
        fig.x_range.end = np.max(x)
        fig.y_range.start = np.min(y)
        fig.y_range.end = np.max(y)
       
        fig.image(image="img",
            source=data_source, 
            x=np.min(x), 
            y=np.min(y), 
            dw=np.max(x)-np.min(x), 
            dh=np.max(y)-np.min(y),
            color_mapper=mapper)
       
        # plot possible polygons on top
        poly_renderer = fig.patches('xs', 'ys', source=polygon_source, 
            fill_color="teal", 
            fill_alpha=0.2,
            line_width=2,
            line_color="black")
       
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
            list_of_polygons = json.loads(f)
    
            # Extract the coordinates of the ROIs
            loaded_data = {"xs": [polygon['x'] for polygon in list_of_polygons],
                           "ys": [polygon['y'] for polygon in list_of_polygons]} 
        
            polygon_source.data = loaded_data
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
    global vline
    global dist_source
    global df
    global is_check_dist

    if is_check_dist:
        if df is not None: 
            target_datetime = pd.to_datetime(event.x, unit='ms')
            diam = df.columns.values.astype(float)
            closest_index = df.index[df.index <= target_datetime].max()
            dist = df.loc[closest_index,:]
            fig_dist.title.text=f'{closest_index}'
            dist_source.data = dict(x=diam, y=dist.values)
            vline.location = int(closest_index.timestamp()*1000.)

def toggle_check_dist(event):
    global is_check_dist
    global vline
    global fig_dist
    global check_dist_button

    if is_check_dist:
        vline.location = np.nan
        dist_source.data = dict(x=[], y=[])
        fig_dist.title.text=f' '
        is_check_dist = False
        check_dist_button.button_type="danger"
    else:
        is_check_dist = True
        check_dist_button.button_type="success"

def close_app():
    sys.exit()

close_js = CustomJS(code="window.close()")

# Variable for the aerosol number-size distribution
df = None

# Variable for the distribution data
dist_source = ColumnDataSource(data=dict(xs=[], ys=[]))

list_of_polygons = []
selected_polygon_index = None

fig = figure(
    width=1000, 
    height=300,
    x_axis_type='datetime',
    x_axis_label="Time",
    y_axis_label="Dp, [m]")

fig_dist = figure(
    width=500,
    height=300,
    x_axis_type="log",
    x_axis_label="Dp, [m]",
    y_axis_label="dN/dlogDp, [cm-3]",
    title = " "
)

fig_dist.line('x', 'y', source=dist_source, line_width=2, color="navy")

is_check_dist = False

vline = Span(location=np.nan, dimension='height', line_color='black', line_width=1, line_dash='dashed')
fig.add_layout(vline)

# Polygon drawing
polygon_source = ColumnDataSource(data=dict(xs=[], ys=[]))

poly_renderer = fig.patches('xs', 'ys', source=polygon_source, 
    fill_color="teal", 
    fill_alpha=0.2,
    line_width=2,
    line_color="black")

draw_tool = FreehandDrawTool(renderers=[poly_renderer])
fig.add_tools(draw_tool)

tap_tool = TapTool(renderers=[poly_renderer])
fig.add_tools(tap_tool)

mapper = LogColorMapper(palette=palette, low=10, high=10000)

data_source = ColumnDataSource(data={'img': []})

color_bar = ColorBar(
    color_mapper=mapper, 
    label_standoff=12, 
    location=(0,0), 
    title="dN/dlogDp [cm-3]")

fig.add_layout(color_bar, 'right')

# Handle the tap tool
fig.on_event('tap',tap_callback)

# Update the selected polygon
poly_renderer.data_source.selected.on_change('indices', update_selected_polygon)

# Update the polygon list
polygon_source.on_change('data', update_list_of_polygons)

# Fit modes
fit_modes_text_input = TextInput(placeholder="Number of Modes",width=310,height=30)

fit_modes_button = Button(label="Fit modes", button_type="success",width=150,height=30)

fit_modes_button.on_event(ButtonClick, fit_modes)

# Load the aerosol data
aerosol_file_button = Button(label="Load aerosol data", button_type="success",height=30,width=150)

aerosol_file_text_input = TextInput(placeholder="Aerosol distribution filepath",height=30,width=310)
       
aerosol_file_button.on_event(ButtonClick, load_aerosol_file)

# Load aerosol data 2
#aerosol_file_input = FileInput(title='Load Aerosol Data')

#aerosol_file_input.on_change('value', load_aerosol_file)

# Update color limits
clim_min_text_input = TextInput(placeholder="Min. value:",value="10",width=150,height=30)

clim_max_text_input = TextInput(placeholder="Max. value:",value="10000",width=150,height=30)

clim_update_button = Button(label="Update color limits", button_type="success",width=150,height=30)
    
clim_update_button.on_event(ButtonClick, update_clim)

# Load ROIs
json_load_button = Button(label="Load ROIs", button_type="success",height=30,width=150)

json_load_text_input = TextInput(placeholder="ROI filepath",height=30,width=310)

json_load_button.on_event(ButtonClick, load_polygon_data)

# Load ROIs
#roi_file_input = FileInput(title='Load ROIs')

#roi_file_input.on_change('value', load_polygon_data)

# Save ROIs
json_save_button = Button(label="Save ROIs", button_type="success",height=30,width=150)

json_save_text_input = TextInput(placeholder="ROI filepath",height=30,width=310)

json_save_button.on_event(ButtonClick, save_polygon_data)

# Check individual distribution button
check_dist_button = Button(label="Check distributions", button_type="danger",height=30,width=150)

check_dist_button.on_event(ButtonClick, toggle_check_dist)

# close the app button
close_app_button = Button(label="Close App", button_type="danger")
close_app_button.js_on_click(close_js)
close_app_button.on_click(close_app)

info_div = Div(text="", width=1000, height=300, styles={"overflow-y": "scroll", "overflow-x": "scroll"})

# Make a layout
layout = column(
    fig,
    row(
        column(
            row(clim_min_text_input,clim_max_text_input,clim_update_button),
            row(aerosol_file_text_input,aerosol_file_button),
            row(json_load_text_input,json_load_button),
            row(json_save_text_input,json_save_button),
            row(fit_modes_text_input,fit_modes_button),
            width=500),
        column(fig_dist,check_dist_button)
    ),
    column(info_div,width=1000),
    close_app_button,
    width=1000
)

# Add the layout to the current document
curdoc().add_root(layout)
