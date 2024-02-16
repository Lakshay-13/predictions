import streamlit as st
import pandas as pd
from prediction import *

# Dummy function for processing data
# Replace or modify this with your actual processing logic
@st.cache_data
def get_tempdata(data, derivatives):
    return get_nn_data(data, derivatives)

def process_data(data, crop_name):
    temp = get_tempdata(data,[0.5, 1, 1.2, 1.5, 2, 2.5, 3])
    results = predict(temp.drop('nm', axis=1).values.tolist(), crop_name)
    return results

# Streamlit UI
def main():
    st.title('Crop Data Analysis')

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        # Assuming the uploaded file is a CSV, load it into a DataFrame
        df = pd.read_csv(uploaded_file, delimiter=";")

        # Crop selection
        # Replace this list with your actual list of crops
        crop_list = ["Limón", "Café", "Maiz", "Naranja", "Uva", "Nogal"]
        selected_crop = st.selectbox('Select a Crop', crop_list)

        # Process data based on selected crop
        if st.button('Show Results'):
            result_df = process_data(df, get_crop_id(selected_crop))
            # Display results
            c1, c2, c3 = st.columns([1,2,1])
            with c2:
                st.text(result_df)

if __name__ == "__main__":
    main()
