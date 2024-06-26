from matplotlib import pyplot as plt
import streamlit as st
from pages.aux import read_files_dict, build_df, plot_distance_differences
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium

##################
############ Main
def run():
    
    """
    Assumptions:
        - files in formtat FIT.gz        
        - Name of the files are the name of the riders
        - Files correspond to the same race 
    """
    
    uploaded_files = st.file_uploader("Choose a set of FIT.gz files", accept_multiple_files=True)
    
    if len(uploaded_files)  == 0 :
        st.write('Please, add fit files"')
    else:

        data = read_files_dict(uploaded_files)

        rides_dict_df = {}
        for name, info in data.items():
            rides_dict_df[name] = build_df(info['ride'], info['events'])

        

        min_timestamp = max(
            [
                df.iloc[0]['timestamp'].timestamp() for name, df in rides_dict_df.items()
            ]
        )

        max_timestamp = max(
            [
                df.iloc[-1]['timestamp'].timestamp() for name, df in rides_dict_df.items()
            ]
        )

        range_values = (datetime.fromtimestamp(min_timestamp), datetime.fromtimestamp(max_timestamp))
        start_datetime = st.slider("choose starting hour", 
                                    min_value=range_values[0], max_value=range_values[1],
                                    value=range_values[0], step=timedelta(minutes=1),
                                    format='H:mm')
        
        rides_corrected = {}
        for name, df in rides_dict_df.items():
            df['valid'] = df['timestamp'].apply(lambda x: x.timestamp() >= start_datetime.timestamp() )

            df_valid = df[df['valid']].reset_index().copy()

            # Correct the starting distance reference
            offset = df_valid.iloc[0]['distance'] 
            df_valid['distance'] = df_valid['distance'].apply(lambda x: x-offset)

            rides_corrected[name] =  df_valid
        ##-
        mean_starting_lat = sum([corrected_df.iloc[0]["position_lat"] for _, corrected_df in rides_corrected.items()])
        mean_starting_lat /= len(rides_corrected)

        mean_starting_long = sum([corrected_df.iloc[0]["position_long"] for _, corrected_df in rides_corrected.items()])
        mean_starting_long /= len(rides_corrected)
        
        map = folium.Map(location=[mean_starting_lat, mean_starting_long ], zoom_start=15)
            
        icons = [ folium.Icon(color="blue", icon="play") for i in range(len(rides_corrected))]
        for i, (rider, data) in enumerate(rides_corrected.items()):
            
            location = (
                data.iloc[0]["position_lat"],
                data.iloc[0]["position_long"],
            )
            folium.Marker(location=location, icon=icons[i]).add_to(map)
            print(rider)
            print(location)
            
            
        st_data = st_folium(map, width=725)


        ##-
        leader = st.selectbox('Select the leader', 
                              tuple(rides_corrected.keys()),
                              index=None,
                              placeholder= "Leader...")
        

        if leader is not None:
            fig, ax = plot_distance_differences(rides_corrected, leader)

            st.pyplot(fig)
        else:
            st.write('Please, select a Valid Leader')


if __name__ == '__main__':
    run()
