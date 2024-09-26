import streamlit as st
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Set the background color using CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #d0f0c0; /* Light green */
        color: black;                /* Default text color */
    }
    h1{
        color: black;                /* Ensure all headings are white */
    }
    .stTextInput label, .stNumberInput label, .stTextArea label, .stSelectbox label {
        color: black;                /* Label color */
    }
    .stButton > button {
        background-color: #4CAF50;  /* Green background for button */
        color: white;                /* White text for button */
    }
    .stButton > button:hover {
        background-color: #45a049;   /* Darker green on hover */
    }
    /* Sidebar styles */
    .stSidebar .stSelectbox label {
        color: black;                /* Label color for sidebar selectbox */
    }
    /* Main page styles */
    .stSelectbox:not(.stSidebar .stSelectbox) label {
        color: black;                /* Label color for main page selectbox */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the data from the CSV file
file_path = "/content/drive/MyDrive/FYP/Agrofood_co2_emission.csv"
df = pd.read_csv(file_path)

# Rename columns to remove spaces and special characters
df.columns = df.columns.str.replace(' ', '_')
df.columns = df.columns.str.replace('-', '')
df.columns = df.columns.str.replace(r'_\(CO2\)', '')
df.columns = df.columns.str.replace(r'_°C', '')

# For numerical columns, use constant strategy to impute missing values with 0
numeric_columns = df.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='constant', fill_value=0)
df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

# Load the pre-trained Random Forest Regressor model
model = joblib.load('/content/drive/MyDrive/FYP/rfr_model.pkl')
scaler = joblib.load('/content/drive/MyDrive/FYP/scaler.pkl')
target_scaler = joblib.load('/content/drive/MyDrive/FYP/target_scaler.pkl')
label_encoder = joblib.load('/content/drive/MyDrive/FYP/label_encoder.pkl')

# Initialize session state if not already
if 'plot_data' not in st.session_state:
    st.session_state.plot_data = None
if 'plot_predicted_data' not in st.session_state:
    st.session_state.plot_predicted_data = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None



def plot_emission_for_countries(df, area_list, emission_type='total_emission'):
    if emission_type not in df.columns:
        st.error(f"Emission type '{emission_type}' does not exist in the DataFrame.")
        return

    plt.figure(figsize=(20, 10))  # Adjust the width and height as needed

    for area_str in area_list:
        filtered_df = df[df['Area'].str.contains(area_str, case=False, na=False)].copy()
        if filtered_df.empty:
            st.warning(f"No matching records found for area string: '{area_str}'")
            continue

        filtered_df['Year'] = pd.to_datetime(filtered_df['Year'], format='%Y')
        filtered_df = filtered_df.sort_values(by='Year')
        plt.plot(filtered_df['Year'], filtered_df[emission_type], marker='o', linestyle='-', label=f"{area_str}")

    plt.xlabel('Year')
    plt.ylabel(f'{emission_type.replace("_", " ").capitalize()} per kiloton (kt)')
    plt.title(f'{emission_type.replace("_", " ").capitalize()} Over Years for Selected Areas')
    plt.legend(title="Areas")
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()

    # st.pyplot(plt)
    # Save the plot to the current figure context
    st.session_state.plot_data = plt.gcf()
    plt.close()

def plot_prediction_graph(df, selected_area, prediction, Current_Year):
    emission_type = 'total_emission'
    if emission_type not in df.columns:
        st.error(f"Emission type '{emission_type}' does not exist in the DataFrame.")
        return

    plt.figure(figsize=(20, 10))

    # Filter data for the selected area
    area_data = df[df['Area'] == selected_area].copy()
    if area_data.empty:
        st.error(f"No data found for the selected area: '{selected_area}'")
        return

    area_data['Year'] = pd.to_datetime(area_data['Year'], format='%Y')
    area_data = area_data.sort_values(by='Year')

    plt.plot(area_data['Year'], area_data[emission_type], marker='o', linestyle='-', label=f"Historical Data for {selected_area}")

    # Convert Current_Year to datetime and plot the predicted value
    predicted_date = pd.to_datetime(f'{Current_Year}', format='%Y')
    plt.scatter(predicted_date, [prediction], color='red', s=100, zorder=5, label='Predicted Value')

    plt.xlabel('Year')
    plt.ylabel(f'{emission_type.replace("_", " ").capitalize()}')
    plt.title(f'{emission_type.replace("_", " ").capitalize()} with Prediction for {selected_area}')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Display the plot with Streamlit
    # st.pyplot(plt)
    st.session_state.plot_predicted_data = plt.gcf()
    plt.close()


# # Streamlit app
# st.title('Emission Data Visualization')

with st.sidebar:
    st.title("Historical data for Emission types")
    emission_type = st.selectbox(
        'Select Emission Type',
        df.columns.difference(['Area', 'Year', 'Savanna_fires', 'Forest_fires', 'Drained_organic_soils_(CO2)', 'Forestland', 'Net_Forest_conversion', 'Fires_in_organic_soils', 'Fires_in_humid_tropical_forests'])
    )

    selected_countries = st.multiselect(
        'Select Countries',
        options=['Philippines', 'Malaysia', 'Thailand', 'Myanmar', "Lao People's Democratic Republic", 'Brunei Darussalam', 'Singapore', 'Viet Nam', 'Indonesia', 'Cambodia']
    )

    if st.button('Plot Emissions'):
        if not selected_countries:
            st.warning('Please select at least one country.')
        else:
            plot_emission_for_countries(df, selected_countries, emission_type)
    st.write("If data remains constant 0 there is no data found within the dataset")

# Initialize session state for each input
input_fields = [
    'Crop_Residues', 'Rice_Cultivation', 'Pesticides_Manufacturing', 'Food_Transport',
    'Food_Household_Consumption', 'Food_Retail', 'Onfarm_Electricity_Use', 'Food_Packaging',
    'Agrifood_Systems_Waste_Disposal', 'Food_Processing', 'Fertilizers_Manufacturing', 'IPPU',
    'Manure_applied_to_Soils', 'Manure_left_on_Pasture', 'Manure_Management', 'Onfarm_energy_use',
    'Rural_population', 'Urban_population', 'Total_Population__Male', 'Total_Population__Female',
    'Average_Temperature'
]

for field in input_fields:
    if field not in st.session_state:
        st.session_state[field] = 0  # Initialize to 0 or other default values

# Create a form for user inputs
st.title('Agriculture Greenhouse Gas Prediction App')

# Collect user inputs
col1, col2 = st.columns(2)
areas = ['Philippines', 'Malaysia', 'Thailand', 'Myanmar', "Lao People's Democratic Republic", 'Brunei Darussalam', 'Singapore', 'Viet Nam', 'Indonesia', 'Cambodia']
selected_area = col1.selectbox('Area', areas, help='Enter the specific area for prediction. Autofill other columns with area specific data')
# Years = list(range(1990, 2021))
# Year = col2.selectbox('Year', Years, help='Enter the specific Year for prediction. Autofill other columns with area specific data')
Year = col2.number_input('Year', step=1, format='%d', value=1990, help='Enter the specific year for prediction. Autofill other columns with year specific data')
Current_Year = Year

# Filter the DataFrame based on the selected Area and Year
filtered_row = df[(df['Area'] == selected_area) & (df['Year'] == Year)]

# If a matching row is found, update session state
if not filtered_row.empty:
    row_data = filtered_row.iloc[0]  # Get the first matching row
    for field in input_fields:
        st.session_state[field] = row_data[field]

# Convert session state values to integers if they are not already
st.session_state['Rural_population'] = int(st.session_state['Rural_population'])
st.session_state['Urban_population'] = int(st.session_state['Urban_population'])
st.session_state['Total_Population__Male'] = int(st.session_state['Total_Population__Male'])
st.session_state['Total_Population__Female'] = int(st.session_state['Total_Population__Female'])

# Display inputs pre-filled with the session state values, users can still modify them
Crop_Residues = col1.number_input('Crop_Residues', value=st.session_state['Crop_Residues'], help='Enter the kilotones of Crop Residues for prediction (Crop residues are waste materials generated by agriculture).')
Rice_Cultivation = col2.number_input('Rice_Cultivation', value=st.session_state['Rice_Cultivation'], help='Enter the kilotones of Rice Cultivation for prediction.')
Pesticides_Manufacturing = col1.number_input('Pesticides_Manufacturing', value=st.session_state['Pesticides_Manufacturing'], help='Enter the kilotones of Pesticides Manufacturing for prediction.')
Food_Transport = col2.number_input('Food_Transport', value=st.session_state['Food_Transport'], help='Enter the kilotones of Food Transport for prediction.')
Food_Household_Consumption = col1.number_input('Food_Household_Consumption', value=st.session_state['Food_Household_Consumption'], help='Enter the kilotones of Food Household Consumption for prediction. (Food household Consumption here looks towards food preperation and wastage)')
Food_Retail = col2.number_input('Food_Retail', value=st.session_state['Food_Retail'], help='Enter the kilotones of Food Retail for prediction.(unsold inventory, refridgerator, etc)')
Onfarm_Electricity_Use = col1.number_input('Onfarm_Electricity_Use', value=st.session_state['Onfarm_Electricity_Use'], help='Enter the On farm Electricity Use for prediction.')
Food_Packaging = col2.number_input('Food_Packaging', value=st.session_state['Food_Packaging'], help='Enter the kilotones of Food Packaging for prediction.')
Agrifood_Systems_Waste_Disposal = col1.number_input('Agrifood_Systems_Waste_Disposal', value=st.session_state['Agrifood_Systems_Waste_Disposal'], help='Enter the kilotones of Agrifood Systems Waste Disposal for prediction. (the waste includes agricultural waste like crop residues and unused produce as well as waste from food manufacturing processes)')
Food_Processing = col2.number_input('Food_Processing', value=st.session_state['Food_Processing'], help='Enter the kilotones of Food Processing for prediction.')
Fertilizers_Manufacturing = col1.number_input('Fertilizers_Manufacturing', value=st.session_state['Fertilizers_Manufacturing'], help='Enter the kilotones of Fertilizers Manufacturing for prediction.')
IPPU = col2.number_input('IPPU', value=st.session_state['IPPU'], help='Enter the industrial processes and product use (IPPU) for prediction. (IPPU in agriculture is related to pesticide and fertilizer manufacturing or refining food into sugar, flour, etc)')
Manure_applied_to_Soils = col1.number_input('Manure_applied_to_Soils', value=st.session_state['Manure_applied_to_Soils'], help='Enter the kilotones of Manure Applied To Soils for prediction.')
Manure_left_on_Pasture = col2.number_input('Manure_left_on_Pasture', value=st.session_state['Manure_left_on_Pasture'], help='Enter the kilotones of Manure Left On Pasture for prediction.')
Manure_Management = col1.number_input('Manure_Management', value=st.session_state['Manure_Management'], help='Enter the kilotones of Manure Management for prediction.')
Onfarm_energy_use = col2.number_input('Onfarm_energy_use', value=st.session_state['Onfarm_energy_use'], help='Enter the Energy Used on the farm for prediction. (electricity, natural gas, coal, oil, biofuels, solar, wind, etc are counted as energy)')
Rural_population = col1.number_input('Rural_population', value=st.session_state['Rural_population'],help='Enter the Rural Population for prediction.')
Urban_population = col2.number_input('Urban_population', value=st.session_state['Urban_population'], help='Enter the Urban Population for prediction.')
Total_Population__Male = col1.number_input('Total_Population__Male', value=st.session_state['Total_Population__Male'], help='Enter the Total Male Population_ for prediction.')
Total_Population__Female = col2.number_input('Total_Population__Female', value=st.session_state['Total_Population__Female'], help='Enter the Total Female Population for prediction.')
Average_Temperature = col1.number_input('Average_Temperature', value=st.session_state['Average_Temperature'], help='Enter the average expected Temperature for prediction.')


# Function to preprocess and predict
def preprocess_and_predict(input_data):
    # if input_data['Year'] > 2020:
      # input_data["Year"] = 2020

    input_df = pd.DataFrame([input_data], columns=[
        'Area', 'Year', 'Crop_Residues', 'Rice_Cultivation',
        'Pesticides_Manufacturing', 'Food_Transport', 'Food_Household_Consumption',
        'Food_Retail', 'Onfarm_Electricity_Use', 'Food_Packaging',
        'Agrifood_Systems_Waste_Disposal', 'Food_Processing',
        'Fertilizers_Manufacturing', 'IPPU', 'Manure_applied_to_Soils',
        'Manure_left_on_Pasture', 'Manure_Management', 'Onfarm_energy_use',
        'Rural_population', 'Urban_population', 'Total_Population__Male',
        'Total_Population__Female', 'Average_Temperature'
    ])

    # Encode Area and normalize
    input_df['Area'] = label_encoder.transform(input_df['Area'])
    input_df = input_df.fillna(input_df.mean())  # Handle missing values
    input_df_normalized = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

    input_df_normalized_float = input_df_normalized.astype(float)

    # Predict and inverse transform the prediction
    prediction_normalized = model.predict(input_df_normalized)
    prediction = target_scaler.inverse_transform(prediction_normalized.reshape(-1, 1))

    return prediction[0][0]

# Button to trigger the prediction
if st.button('Predict Total Emission'):
    # Prepare the feature list for prediction
    input_values = {
        'Area': selected_area,  # Make sure this value matches one of the encoded labels
        'Year': Year,
        'Crop_Residues': Crop_Residues,
        'Rice_Cultivation': Rice_Cultivation,
        'Pesticides_Manufacturing': Pesticides_Manufacturing,
        'Food_Transport': Food_Transport,
        'Food_Household_Consumption': Food_Household_Consumption,
        'Food_Retail': Food_Retail,
        'Onfarm_Electricity_Use': Onfarm_Electricity_Use,
        'Food_Packaging': Food_Packaging,
        'Agrifood_Systems_Waste_Disposal': Agrifood_Systems_Waste_Disposal,
        'Food_Processing': Food_Processing,
        'Fertilizers_Manufacturing': Fertilizers_Manufacturing,
        'IPPU': IPPU,
        'Manure_applied_to_Soils': Manure_applied_to_Soils,
        'Manure_left_on_Pasture': Manure_left_on_Pasture,
        'Manure_Management': Manure_Management,
        'Onfarm_energy_use': Onfarm_energy_use,
        'Rural_population': Rural_population,
        'Urban_population': Urban_population,
        'Total_Population__Male': Total_Population__Male,
        'Total_Population__Female': Total_Population__Female,
        'Average_Temperature': Average_Temperature
    }

    # Predict using the model with the user inputs
    prediction = preprocess_and_predict(input_values)
    st.session_state.prediction = prediction

    # Plot the prediction graph
    plot_prediction_graph(df, selected_area, prediction, Current_Year)

    # # Display the prediction result
    # st.write(f'Predicted Total Emission: {st.session_state.prediction:.2f}')

# Display the plot in the sidebar if available
if st.session_state.plot_data is not None:
    with st.sidebar:
        st.pyplot(st.session_state.plot_data)

# Display the plot in the sidebar if available
if st.session_state.plot_predicted_data is not None:
        st.pyplot(st.session_state.plot_predicted_data)

if st.session_state.prediction is not None:
    st.write(f'Predicted Total Emission: {st.session_state.prediction:.2f}')
    
