import streamlit as st


##################
############ Main
def run():
    st.write("Add files. Each file should have as name the name of the rider")

    uploaded_files = st.file_uploader("Choose a set of FIT.gz file for the first rider", accept_multiple_files=True)


if __name__ == '__main__':
    run()
