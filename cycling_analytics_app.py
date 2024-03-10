import streamlit as st 
from garmin_fit_sdk import Decoder, Stream
import gzip
import pandas as pd
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

### CONSTANTS
TEST = True

# Number of decimals for rounding in the coordinates
N_DECIMALS = 4

#Difference in seconds to consider two activities correspond to the same race
DELTA_SECONDS = 2 # 1 minutes

# Speed threshold, in Km/h to consider two speeds as equal
SPEED_THR = 0.25
### END CONSTANTS


### Functions
@st.cache_data
def read_files(list_file_content):
    rides = []
    events = []
    for f in list_file_content:
        bytes_data = f.read()

        fit_raw_data = gzip.decompress(bytes_data)

        
        #TODO: This code gives problems with file Malcotti, strade Bianche. Find out why
        
        stream = Stream.from_byte_array(fit_raw_data)
        decoder = Decoder(stream)
        messages, errors = decoder.read()
        
        if len(messages['record_mesgs'])>3:
            rides.append(messages['record_mesgs'])
            events.append(messages['event_mesgs'])

    return rides, events

@st.cache_data
def build_df(record_msg, events_msg):
    #st.write('build_df')
    df = pd.DataFrame(record_msg)

    df['is_start'] = False
    


    # values without gps coordinates are uselss
    df.dropna(subset=["timestamp", "position_long", "position_lat", "distance", "enhanced_speed"],inplace=True)
    df.reset_index(drop=True, inplace=True)


    # Transform into degrees
    df["position_long"] = (df["position_long"] / 11930465).round(N_DECIMALS)
    df["position_lat"] = (df["position_lat"] / 11930465).round(N_DECIMALS)
    
    # Transorm from m/s to Km/h
    df["speed"] = (df['enhanced_speed']*3.6).round(1)

    #Distance in meters
    df["distance"] = (df["distance"] ).round().astype(int)

    start_hours = [ msg['timestamp'].timestamp() for msg in events_msg if ( msg['event']=='timer' and msg['event_type']=='start')]

    df['is_start'] = df['timestamp'].apply(lambda x : x.timestamp() in start_hours)
    df.loc[0, 'is_start'] = True



    return df


@st.cache_data
def pair_rides(list_file_content_1, list_file_content_2):
    #st.write('pair_rides')

    rides_1, events_1 = read_files(list_file_content_1)
    rides_2, events_2 = read_files(list_file_content_2)

    rides_df_1 = [build_df(r, e) for r,e in zip(rides_1, events_1)]
    rides_df_2 = [build_df(r, e) for r,e in zip(rides_2, events_2)]

    paired_1 = []
    paired_2 = []

    for df_1 in rides_df_1:        
        i = 0
        equals = False
        while i < len(rides_df_2):
            df_2 = rides_df_2[i]

            if df_1.shape[0] > 2 and df_2.shape[0] > 2:
                diff_time = (max(df_1.iloc[2]['timestamp'], df_2.iloc[2]['timestamp']) - min(df_1.iloc[2]['timestamp'], df_2.iloc[2]['timestamp']))
                #diff_pos = max(df_1.iloc[2]['position_long'] - df_2.iloc[2]['position_long'], df_1.iloc[2]['position_lat'] - df_2.iloc[2]['position_lat'])

                #if races are the same day
                if  diff_time.days == 0:
                    
                    paired_1.append(df_1)
                    paired_2.append(df_2)
            
            i += 1

    return paired_1, paired_2
        
@st.cache_data        
def get_joint_data(df_1, df_2, start_hour, tracks = None):
    #st.write('get_joint_data')

    def set_start(df):
        located_start = False
        pos_time = 0
    
        while (not located_start) and (pos_time < df.shape[0]):
            t = df.iloc[pos_time]['timestamp']
        
            if abs(start_hour - t.timestamp()) <= 1:
                df = df[pos_time:-1]
                
                offset = df.iloc[0]['distance'] 
                df['distance'] = df['distance'].apply(lambda x: x-offset)                                                

                located_start = True
        
            pos_time +=1

        return df
    
    df_1 = set_start(df_1)
    df_2 = set_start(df_2)

    end = int(np.round((df_1.iloc[-1]["distance"]+ df_2.iloc[-1]["distance"])/2))

    distances = list(range(end))

    distances_1 = df_1["distance"].values
    distances_2 = df_2["distance"].values

    to_interpolate_1 = np.array(list(range(end)))
    to_interpolate_2 = np.array(list(range(end)))
    
    vel_1 = df_1["speed"].values
    vel_2 = df_2["speed"].values

    altitud_1 = df_1["enhanced_altitude"].values
    altitud_2 = df_2["enhanced_altitude"].values

    speeds_1 = np.interp(to_interpolate_1, distances_1, vel_1)
    speeds_2 = np.interp(to_interpolate_2, distances_2, vel_2)

    
    altitud_1_interp = np.interp(to_interpolate_1, distances_1, altitud_1)
    altitud_2_interp = np.interp(to_interpolate_2, distances_2, altitud_2)


    altitudes = (altitud_1_interp + altitud_2_interp)/2


    speeds_1=signal.savgol_filter(speeds_1,
                           60, # window size used for filtering
                           2) # order o

    speeds_2=signal.savgol_filter(speeds_2,
                           60, # window size used for filtering
                           2) # order o

    return speeds_1, speeds_2, distances, altitudes

# TODO: adapt it to have dataframes
@st.cache_data
def plot_profile_comparative(speeds_1, speeds_2, distances, altitudes, name_1, name_2, title):
    #st.write('plot_profile_comparative')

    # to km
    distances = distances /1000

    fig, ax = plt.subplots()
    fig.set_size_inches(20, 8)
    
    diff = speeds_1 - speeds_2

    x = distances[diff>=SPEED_THR] 
    y = altitudes[diff>=SPEED_THR]    
    ax.scatter(x, y, s= 10,color='blue', label =f"{name_1}" )
    #plt.scatter(distances, altitudes)

    x = distances[diff <= -SPEED_THR]
    y = altitudes[diff <= -SPEED_THR]
    ax.scatter(x, y, s= 10,color='red', label =f"{name_2} " )

    idx = (diff > -SPEED_THR) & (diff < SPEED_THR)
    ax.scatter(distances[idx], altitudes[idx], s= 5, color='yellow', label = f"Equal")
    
    ax.legend(fontsize=25, loc='best')
    ax.set_title(title, fontsize=25)
    ax.set_xlabel( 'Km', fontsize=25)
    ax.set_ylabel('Altitude', fontsize=25)
    #ax.set_xticklabels(distances[idx], fontsize=25)
    #ax.set_yticklabels(altitudes[idx], fontsize=25)

    return fig

@st.cache_data
def locate_start(df_1, df_2):
    start = None
    starts_1 = []
    starts_2 = []

    start_hours = df_1[df_1['is_start']]['timestamp']

    
    for  time_1 in start_hours:

        located_start = False
        pos_2 = 0
        while (not located_start) and (pos_2 < df_2.shape[0] ):
            time_2 = df_2.iloc[pos_2]['timestamp']

            if abs(time_1.timestamp() - time_2.timestamp() ) < DELTA_SECONDS:
                located_start = True

                starts_1.append(time_1)
        
            pos_2 += 1

 
    start_hours = df_2[df_2['is_start']]['timestamp']    
    for  time_2 in start_hours:

        located_start = False
        pos_1 = 0
        while (not located_start) and (pos_1 < df_1.shape[0] ):
            time_1 = df_1.iloc[pos_1]['timestamp']

            if abs(time_1.timestamp() - time_2.timestamp() ) < DELTA_SECONDS:
                located_start = True

                starts_2.append(time_2)
        
            pos_1 += 1


    # Keep the lowest
    #located_start = False
    #pos_1 = 0
    # while (not located_start) and (pos_1 < len(starts_1)):
    #    time_1 = starts_1[pos_1]

    #    pos_2 = 0
    #    while (not located_start) and (pos_2 < len(starts_2)):
    #        time_2 = starts_2[pos_2]

    start = min (starts_1 + starts_2)

    return start


# def make_map(df_1, df_2):
     
#         speeds_1 = df_1["speed"].values
#         speeds_2 = df_2["speed"].values

#         # Speed: 
#         # -1: df_2 faster
#         # 0: equal
#         # +1: df_1 faster
#         speed = np.zeros_like(speeds_1)
        
#         diff = speeds_1 - speeds_2
        
#         speed[diff<= -SPEED_THR] = -1

#         speed[diff>=SPEED_THR] = 1

#         speed[0] = -1
#         speed[1] = 0
#         speed[2] = 1

#         #colormap = branca.colormap.linear.viridis.scale(0,50)
#         colormap = ["red", "yellow", "blue"]
#         map = folium.Map(location=[df_1.iloc[10]["position_lat"], df_1.iloc[10]["position_long"] ], zoom_start=13)
#         track = [ (lat, long) for lat,long in zip(df_1["position_lat"].values, df_1["position_long"].values)]
        
#         folium.ColorLine(positions = track, # tuple of coordinates 
#                          colors = speed, # map each segment with the speed 
#                          colormap =  colormap, # map each value with a color 
#                          weight=5
#                          ).add_to(map)
        
#         return map



##################
############ Main
def run():
    ### Streamlit
    rider_name_1 = st.text_input('Name of the first rider')
    
    
    uploaded_files_1 = st.file_uploader("Choose a set of FIT.gz file for the first rider", accept_multiple_files=True)

    rider_name_2 = st.text_input('Name of the second rider')

    uploaded_files_2 = st.file_uploader("Choose a set of FIT.gz file for the second rider", accept_multiple_files=True)

    rides_df_1, rides_df_2= pair_rides(uploaded_files_1, uploaded_files_2)

    if len(rides_df_1)  == 0 or (len(rides_df_1) != len(rides_df_2)):
        st.write('THERE IS NO CORRESPONDENCE BETWEEN FILES')
    else:        

        for df_1, df_2 in zip (rides_df_1, rides_df_2):
            start_datetime = locate_start(df_1, df_2)

            start_date = st.date_input('Estimated day ', value=start_datetime)
            start_time = st.time_input("Estimated hour, fix it if needed: ", value = start_datetime, step=60)

            start_datetime = datetime.combine(start_date, start_time)
            start_datetime = datetime.timestamp(start_datetime)

            s1, s2, dis, alt= get_joint_data(df_1, df_2, start_datetime)

            str_date = df_1.iloc[0]['timestamp'].strftime("%d %B, %Y")
            title = f"Race comparative: {str_date}"

            fig = plot_profile_comparative(s1, s2, np.array(dis), np.array(alt), rider_name_1, rider_name_2, title)

            st.pyplot(fig)

            #map = make_map(df_1, df_2)


            #st_folium(map, width=725)


if __name__ == '__main__':
    run()



