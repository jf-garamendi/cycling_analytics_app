from matplotlib import pyplot as plt
import streamlit as st 
import pandas as pd
import gzip
from garmin_fit_sdk import Decoder, Stream
from datetime import datetime, timedelta

N_DECIMALS = 3

### Function

# def plot_distance_differences(dataframes_dict, leader):
#     fig, ax = plt.subplots()
#     fig.set_size_inches(20, 8)

#     # Initialize an empty DataFrame to store the differences
#     differences_df = pd.DataFrame(columns=['timestamp', 'difference'])

#     reference_df = dataframes_dict[leader]

#     # Iterate through each dataframe in the list
#     for name, df in dataframes_dict.items():
#         # Iterate through each row in the current dataframe
#         for index, row in df.iterrows():
#             # Find the nearest timestamp in the reference dataframe
#             nearest_time = reference_df.iloc[
#                 (reference_df['timestamp'] - row['timestamp']).abs().argsort()[:1]
#                 ]['timestamp'].values[0]
            
#             # Find the corresponding row in the reference dataframe
#             reference_row = reference_df[reference_df['timestamp'] == nearest_time]
            
#             # Calculate the difference in the "distance" column between the current dataframe and the reference dataframe
#             difference = row['distance'] - reference_row['distance'].values[0]
            
#             # Append the difference to the differences DataFrame
#             differences_df = differences_df.append({'timestamp': row['timestamp'], 'difference': difference}, ignore_index=True)

#         # Plot the differences against the "time" column
#         ax.plot(differences_df['timestamp'], differences_df['difference'], label= name)

#         # Add labels and title
#         ax.set_xlabel('Time')
#         ax.set_ylabel('Distance Difference')
#         ax.set_title('Distance Difference vs Time')
#         ax.legend()

#     return fig, ax


def plot_distance_differences(dataframes_dict, leader):
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 8)

    # Initialize an empty DataFrame to store the differences
    differences_df = pd.DataFrame(columns=['timestamp', 'difference'])

    reference_df = dataframes_dict[leader]


    # Iterate through each dataframe in the list
    for name, df in dataframes_dict.items():
        max_lenght = min(df.shape[0], reference_df.shape[0])
        differences = reference_df.iloc[:max_lenght]['distance'].values - df.iloc[:max_lenght]['distance'].values
        
        # Plot the differences against the "time" column
        ax.plot(reference_df.iloc[:max_lenght]['distance'], differences, label= name)

        # Add labels and title
        ax.set_xlabel('Distance')
        ax.set_ylabel('Distance Difference')
        ax.set_title('Distance Vs Difference')
        ax.legend()
    
    return fig, ax





@st.cache_data
def read_files_dict(list_file_content):
    data = {}
    for f in list_file_content:
        filename = f.name
        bytes_data = f.read()

        fit_raw_data = gzip.decompress(bytes_data)
   
        stream = Stream.from_byte_array(fit_raw_data)
        decoder = Decoder(stream)
        messages, errors = decoder.read()
        
        if len(messages['record_mesgs'])>3:
            data[filename]={
                'ride': messages['record_mesgs'],
                'events': messages['event_mesgs'] 
            }

    return data


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
    df["position_long"] = (df["position_long"] / 11930465) 
    df["position_lat"] = (df["position_lat"] / 11930465)

    df["rounded_long"] = df["position_long"].round(N_DECIMALS)
    df["rounded_lat"] = df["position_lat"].round(N_DECIMALS)
    
    # Transorm from m/s to Km/h
    df["speed"] = (df['enhanced_speed']*3.6).round(1)

    #Distance in meters
    df["distance"] = (df["distance"] ).round().astype(int)

    start_hours = [ msg['timestamp'].timestamp() for msg in events_msg if ( msg['event']=='timer' and msg['event_type']=='start')]

    df['is_start'] = df['timestamp'].apply(lambda x : x.timestamp() in start_hours)
    df.loc[0, 'is_start'] = True



    return df

@st.cache_data
def add_kilojoules_per_hour(df):
    # Convert power in watts to kilojoules (1 watt-second = 0.001 kilojoules)
    df['kilojoules'] = df['power'] * 0.001
    

    df['kilojoules_last_hour']  = df['kilojoules'].rolling(3600).sum()
    
    return df

#@st.cache_data
def add_best_power_values(df, period_list):
    #period_list in seconds, e.g, [30, 60, 600, 1200, 3600] for best 30", 1', 10', 20', 60'
    # Calculate the rolling sum over the last 3600 seconds (1 hour), assuming the data is in 1-second intervals
    for period in period_list:
        df[f'Best {period}"']  = df['power'].rolling(period).mean() 
    
    return df



@st.cache_data
def plot_df(df, x_column='distance', y_columns=['altitude', 'kilojoules'], fig=None, ax = None):

    if fig is None and ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 8)

    ax.plot(
        df['distance'],
        df['altitude']
        )

    return fig, ax