from matplotlib import pyplot as plt
import streamlit as st
from pages.aux import read_files_dict, build_df, add_best_power_values,  compute_avg_NP, add_cs
from datetime import datetime, timedelta
import pandas as pd
import io
from datetime import timedelta



def show_metrics(riders_data, weights, ftps):
    data = []
    for rider, df in riders_data.items():
        # Build the best poers
        ftp = ftps[rider]
        weight = weights[rider]
        df = add_best_power_values(df, [30, 60, 600, 1200, 3600])
        df = add_cs(df, [1, 5, 12])

        NP = compute_avg_NP(df)

        df['above FTP'] = df['power'].apply(lambda x: 1 if x > ftp else 0)

        AP_FTP = (df['above FTP'].sum() / df.shape[0]) * 100

        #computing coasting
        df['w is 0'] = df['power'].apply(lambda x : 1 if x <= 30 else 0)

        coasting = (df['w is 0'].sum() / df.shape[0]) * 100

        duration = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]

        hours = duration.seconds // 3600

        minutes = (duration.seconds - hours*3600) // 60

        seconds = duration.seconds - hours*3600 - minutes * 60

        duration_str = f'{hours}:{minutes:02d}:{seconds:02d}'


        row = {
            'name': rider,
            'time': duration_str,
            'Pos': None,
            'Coasting': coasting,
            'distance': df['distance'].iloc[-1]/1000,
            'Avg Speed': df['speed'].mean(),            
            'Avg Power': df['power'].mean(),
            'NP': NP,
            'IF': NP/ftp,
            'AP  FTP': AP_FTP,
            'Work (Kj)': df['power'].sum()*0.001,
            'Power/kg': df['power'].mean()/weight,
            'NP/kg': NP/weight,
            'Kj/kg': df['power'].sum()*0.001 / weight,
            'Pmax' : None,
            'Best 30" ': df['Best 30"'].max(),
            "Best 1'  ": df['Best 60"'].max(),
            "Best 10' ": df['Best 600"'].max(),
            "Best 20' ": df['Best 1200"'].max(),
            "Best 60' ": df['Best 3600"'].max(),
            "CS 1' ": (df['cs 1'].max()**2) / weight,
            "CS 5' ": (df['cs 5'].max()**2)/weight,
            "CS 12' ": (df['cs 12'].max()**2) / weight,
            'Avg HR': df['heart_rate'].mean()
        }

        for k, v in row.items():
            if k != 'name' and k != 'time' and v is not None:
                row[k] = round(v, 2)

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
            ftp = st.number_input(f'FTP {name}', key=f'ftp_{name}', value=1 )
            w = st.number_input(f'Weight {name}', key=f'weight_{name}', value=1)

            ftp_dict[name] = ftp
            weight_dict[name] = w



        st.header("Result")

        show_metrics(rides_corrected, weight_dict, ftp_dict)



if __name__ == '__main__':
    run()
