from matplotlib import pyplot as plt
import streamlit as st
from pages.aux import read_files_dict, build_df, add_best_power_values,  compute_avg_NP
from datetime import datetime, timedelta
import pandas as pd
import io
from datetime import timedelta



def show_metrics(riders_data, weights, ftps):
    data = []
    for rider, df in riders_data.items():
        # Build the best poers
        df = add_best_power_values(df, [30, 60, 600, 1200, 3600])

        NP = compute_avg_NP(df)

        #computing coasting
        df['w is 0'] = df['power'].apply(lambda x : 1 if x <= 10 else 0)

        coasting = df['w is 0'].sum() / df.shape[0]

        row = {
            'name': rider,
            'Pos': None,
            'Coasting': coasting,
            'distance': df['distance'].iloc[-1]/1000,
            'Avg Speed': df['speed'].mean(),            
            'Avg Power': df['power'].mean(),
            'NP': NP,
            'IF': NP/ftps[rider],
            'AP  FTP': None,
            'Work (Kj)': df['power'].sum()*0.001,
            'Power/kg': df['power'].mean()/weights[rider],
            'NP/kg': NP/weights[rider],
            'Kj/kg': df['power'].sum()*0.001 / weights[rider],
            'Pmax' : None,
            'Best 30" ': df['Best 30"'].max(),
            "Best 1'  ": df['Best 60"'].max(),
            "Best 10' ": df['Best 600"'].max(),
            "Best 20' ": df['Best 1200"'].max(),
            "Best 60' ": df['Best 3600"'].max(),
            "CS 1' ": None,
            "CS 5' ": None,
            "CS 12' ": None,
            'Avg HR': df['heart_rate'].mean()
        }

        data.append(row)

    df2show = pd.DataFrame(data)
    st.dataframe(df2show)

    buffer = io.BytesIO()

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Write each dataframe to a different worksheet.
        df2show.to_excel(writer, sheet_name='Sheet1')


    st.download_button(
        label="Download Excel worksheets",
        data=buffer,
        file_name="comparative.xlsx",
        mime="application/vnd.ms-excel"
    )

        




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

        

        min_timestamp = min(
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

        st.header("Add FTP and weight")
        weight_dict = {}
        ftp_dict = {}
        for name in rides_corrected.keys():
            ftp = st.number_input(f'FTP {name}', key='ftp_{name}', value=1)
            w = st.number_input(f'Weight {name}', key='weight_{name}', value=1)

            ftp_dict[name] = ftp
            weight_dict[name] = w



        st.header("Result")

        show_metrics(rides_corrected, weight_dict, ftp_dict)



if __name__ == '__main__':
    run()
