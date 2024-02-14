from spectra import *
import pandas as pd
import differint.differint as dif
from prettytable import PrettyTable
from joblib import Parallel, delayed


def _apply_gli(order, nm, percent_r):
    return dif.GLI(order, percent_r, domain_start=min(nm), domain_end=max(nm), num_points=len(nm))

# Function to process the data and create new columns for each order
def get_nn_data(df, p):
  
    # Retrieve the 'nm' column from the combined DataFrame and get the first element's list
    nm = list(df["nm"])
    r = list(df['%R'])

    # Nested helper function that applies the GLI function to the '%R' column
    # This will be the function that gets parallelized for each 'order' in 'p'
    def process_order(order):
        # Apply the '_apply_gli' function to each row's '%R' value
        return  _apply_gli(order, nm, r)
        
    # Execute the 'process_order' function in parallel for each element in 'p'
    # 'n_jobs=-1' indicates that all available CPU cores should be used for parallelization
    results = Parallel(n_jobs=-1)(delayed(process_order)(order) for order in p)

    # Iterate over the 'order' and corresponding 'result'
    # to create a new column in 'combined_df' for each 'order'
    for order, result in zip(p, results):
        df[f'%R {order}'] = result

    # Return the modified DataFrame with new columns added
    return pd.DataFrame({col: [df[col].tolist()] for col in df.columns})

def predict(sample, crop_name):
    units = {'N':'%', 'P':'%', 'K':'%', 'Ca':'%', 'Mg':'%', 'Fe':'mg/kg', 'Cu':'mg/kg', 'Zn':'mg/kg', 'Mn':'mg/kg', 'B':'mg/kg'}
    base_model = tf.keras.models.load_model(fr"Models/Model/Base_Model")
    crop_model = tf.keras.models.load_model(fr'Models/Model/Crops/{crop_name}/{crop_name}_model')
    base_result = base_model.predict(sample)
    crop_result = crop_model.predict(sample)
    table = PrettyTable()
    table.field_names = ["Element", "Base Model", "Crop Model"]
    for element in units.keys():
        ele = f"{element} [{units[element]}]"
        table.add_row([ele, f'{base_result[element][0][0]:.2f}', f'{crop_result[element][0][0]:.2f}'])
    return table

def get_crop_name(x):
    ['pi', 'lim', 'caf', 'ma', 'naranja', 'uva', 'nogal']
    ["Piña", "Limón", "Café", "Maiz", "Naranja", "Uva", "Nogal"]
    if x == "pi":
        return "Piña"
    elif x == "lim":
        return "Limón"
    elif x == "caf":
        return "Café"
    elif x == "ma":
        return "Maiz"
    elif x == "naranja":
        return "Naranja"
    elif x == "uva":
        return "Uva"
    elif x == "nogal":
        return "Nogal"
    
def get_crop_id(x):
    if x == "Piña":
        return "pi"
    elif x == "Limón":
        return "lim"
    elif x == "Café":
        return "caf"
    elif x == "Maiz":
        return "ma"
    elif x == "Naranja":
        return "naranja"
    elif x == "Uva":
        return "uva"
    elif x == "Nogal":
        return "nogal"