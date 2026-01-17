import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Importing model and data
pipe = pickle.load(open('pipe2.pkl', 'rb'))
df = pickle.load(open('df2.pkl', 'rb'))

# Initialize label encoder for categorical columns
brand_encoder = LabelEncoder()
cpu_encoder = LabelEncoder()
os_encoder = LabelEncoder()
gpu_type_encoder = LabelEncoder()
# Fit the label encoders on the unique values in the dataframe
brand_encoder.fit(df['brand'])
cpu_encoder.fit(df['processor'])
os_encoder.fit(df['os'])
gpu_type_encoder.fit(df['Gpu_brand'])

st.title('Laptop Price Prediction')

# Dropdown for selecting the features
Brand = st.selectbox('Brand', df['brand'].unique())
type = st.selectbox('Type', df['name'].unique())
ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
ram_type = st.selectbox('RAM_TYPE(LPDDR/UNIFIED)', ['No', 'Yes'])
warranty = st.selectbox('Warranty(in yrs)', df['warranty'].unique())
gpu = st.selectbox('GPU', ['No', 'Yes'])
screen_size = st.number_input('Screen Size(in inch)')
resolution = st.selectbox('Resolution', [
    '1920x1080', '2560x1440', '1366x768', '2880x1800', '3840x2160',
    '3200x1800', '1080x1920', '1600x1200', '3024x2464', '1200x1080',
    '3456x2160', '2160x1440', '2240x1760', '2496x1440', '1280x1024',
    '2256x1504', '3072x2048', '1440x900', '2560x1600'
])
cpu = st.selectbox('CPU', df['processor'].unique())
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1000, 2000])
ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])
gpu_type = st.selectbox('GPU_Type', df['Gpu_brand'].unique())
os = st.selectbox('OS', df['os'].unique())
no_of_cores = st.selectbox('No_of_cores', df['Number_of_cores'].unique())

if st.button('Predict Price'):
    # Preprocess query input
    ppi = None
    if gpu == 'Yes':
        gpu = 1
    else:
        gpu = 0

    # Map 'ram_type' based on the selection
    if ram_type == 'Yes':
        ram_type = 0
    else:
        ram_type = 1

    # Split resolution into X and Y
    X_resolution, Y_resolution = map(int, resolution.split('x'))

    # Calculate ppi (pixels per inch)
    ppi = (((X_resolution ** 2) + (Y_resolution ** 2)) ** 0.5) / screen_size

    # Manually encode categorical variables
    Brand_encoded = brand_encoder.transform([Brand])[0]
    CPU_encoded = cpu_encoder.transform([cpu])[0]
    OS_encoded = os_encoder.transform([os])[0]
    GPU_type_encoded = gpu_type_encoder.transform([gpu_type])[0]

    # Prepare the query as a numerical array
    query = np.array([
        Brand_encoded, type, ram, ram_type, gpu, screen_size, ppi, CPU_encoded, hdd, ssd, GPU_type_encoded, OS_encoded,
        no_of_cores, warranty
    ])

    # Reshape the query to be a 2D array for the model
    query = query.reshape(1, -1)

    # Predict using the pipeline and display the result
    prediction = (np.exp(pipe.predict(query)))

    # Display the result
    st.title(f'Predicted Laptop Price: {prediction[0]*2.4:,.2f}')
