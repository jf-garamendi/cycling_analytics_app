# from matplotlib import pyplot as plt
import streamlit as st
# from aux import read_files, build_df, add_kilojoules_per_hour
# from datetime import datetime, timedelta

# ##################
# ############ Main
# def run():
#     """
#     Assumptions:
#         - files in formtat FIT.gz        
#         - Name of the files are the name of the riders
#         - Files correspond to the same race 
#     """

#     uploaded_files = st.file_uploader("Choose a set of FIT.gz files", accept_multiple_files=True)

#     if len(uploaded_files)  == 0 :
#         st.write('Please, add fit files"')
#     else:

#         rides, events = read_files(uploaded_files)

#         rides_df= [build_df(r, e) for r,e in zip(rides, events)]

#         min_timestamp = min(
#             [
#                 df.iloc[0]['timestamp'].timestamp() for df in rides_df
#             ]
#         )

#         max_timestamp = max(
#             [
#                 df.iloc[-1]['timestamp'].timestamp() for df in rides_df
#             ]
#         )

#         range_values = (datetime.fromtimestamp(min_timestamp), datetime.fromtimestamp(max_timestamp))
#         start_datetime = st.slider("choose starting hour", 
#                                     min_value=range_values[0], max_value=range_values[1],
#                                     value=range_values[0], step=timedelta(minutes=1),
#                                     format='H:mm')
        
#         rides_corrected = []
#         for df in rides_df:
#             df['valid'] = df['timestamp'].apply(lambda x: x.timestamp() >= start_datetime.timestamp() )

#             df_valid = df[df['valid']].reset_index().copy()

#             # Correct the starting distance reference
#             offset = df_valid.iloc[0]['distance'] 
#             df_valid['distance'] = df_valid['distance'].apply(lambda x: x-offset)                                                

#             # Compute Kilojules
#             df_valid = add_kilojoules_per_hour(df_valid)

#             rides_corrected.append(df_valid)


#         fig, ax = plt.subplots()
#         fig.set_size_inches(20, 8)

#         for df in rides_corrected:
#             ax.set_ylabel('Kj/h/kg', fontsize=25)
#             ax.plot(df['distance']/1000, df['kilojoules_last_hour'], color='blue')



#         fig, ax_altitude = plot_profile_comparative(s1, s2, np.array(dis), np.array(alt), rider_name_1, rider_name_2, title)


#             ax2 = ax_altitude.twinx()
#             ax2.set_ylabel('Kj/h/kg', fontsize=25)
            
#             ax2.plot(df_1_bis['distance']/1000, df_1_bis['kilojoules_last_hour']/weight_1, color='blue')
#             ax2.plot(df_2_bis['distance']/1000, df_2_bis['kilojoules_last_hour']/weight_2, color='red')


#             st.pyplot(fig)



if __name__ == '__main__':
#     run()
    st.title("Work in progress")
