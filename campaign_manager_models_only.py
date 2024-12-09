import streamlit as st
import pandas as pd # version 2.0.3
import numpy as np
import plotly.express as px # version 5.15.0
import matplotlib.pyplot as plt # version 3.7.1
import matplotlib.ticker as mtick
import joblib
from sklearn.ensemble import RandomForestClassifier
from google.cloud import storage
from google.oauth2 import service_account
import io


# set up wide page on streamlit app
st.set_page_config(layout='centered', initial_sidebar_state='expanded')

##########################################
# load file for models from Google Cloud #
##########################################
# Load file anon_processed_unique_device_v3.csv from GCP
print("I love you!"
# Set up Google Cloud credentials using secrets stored in Streamlit Cloud
# service_key = {
#   "type": "service_account",
#   "project_id": st.secrets['PROJECT_ID'],
#   "private_key_id": st.secrets['PRIVATE_KEY_ID'],
#   "private_key": st.secrets['PRIVATE_KEY'],
#   "client_email": st.secrets['CLIENT_EMAIL'],
#   "client_id": st.secrets['CLIENT_ID'],
#   "auth_uri": st.secrets['AUTH_URI'],
#   "token_uri": st.secrets['TOKEN_URI'],
#   "auth_provider_x509_cert_url": st.secrets['AUTH_PROVIDER_URL'],
#   "client_x509_cert_url": st.secrets['CLIENT_CERT_URL'],
#   "universe_domain": "googleapis.com"
# }

# #Downloaded credentials in JSON format
# project_id=service_key["project_id"]
# credentials = service_account.Credentials.from_service_account_info(service_key)
# client = storage.Client(project=project_id,credentials=credentials)

# # Access the file in the bucket
# bucket_name = 'campaign_manager_tool'
# file_name = 'anon_processed_unique_device_v3.csv'
# bucket = client.bucket(bucket_name)
# blob = bucket.blob(file_name)

# # Download the file content and read as a dataframe
# content = blob.download_as_bytes()
# anon_df = pd.read_csv(io.BytesIO(content))

# print(anon_df)

# ############################
# # Load Datasets for Models #
# ############################
# @st.cache_data
# def load_data():
#     engagement_data = anon_df
#     campaign_data = pd.read_csv('data/anan_campaign_modeling_data_v3.csv')
#     return engagement_data, campaign_data

# # Load Models for Prediction Tools
# @st.cache_resource
# def load_models():
#   with st.spinner('Loading model...'):
#     reg_model = joblib.load(r'models/regression_model.pkl')
#     rf_classifier = joblib.load(r'models/rf_classifier.pkl')
#     nn_model = joblib.load(r'models/nearest_neighbors_model.pkl')
#     scaler = joblib.load(r'models/nearest_neighbors_scaler.pkl')
#   return reg_model, rf_classifier, nn_model, scaler

# # Preprocess Engagement Data
# @st.cache_data
# def preprocess_engagement_data(data):
#     data['event_date'] = pd.to_datetime(data['event_date'].str[12:22])
#     data['install_date'] = pd.to_datetime(data['install_date'].str[12:22])
#     return data

# # Load Data and Models
# engagement_data, campaign_data = load_data()
# engagement_data = preprocess_engagement_data(engagement_data)
# reg_model, rf_classifier, nn_model, scaler = load_models()

# # Columns for Genre Input
# genre_columns = [
#     "Classics", "Comedy", "Other", "Reality", "News and Information",
#     "Drama", "Action & Adventure", "Thriller", "Sci-Fi & Fantasy", "Horror", "Western",
#     "Documentaries", "Sports", "Instructional & Educational", "Home & Lifestyle", "Romance",
#     "Anime", "Musical", "Independent", "Entertainment", "Paranormal", "Music", "Gay & Lesbian",
#     "Crime", "Food & Cooking", "Faith & Spirituality", "Game Show", "Dance", "Children & Family",
#     "Telenovela", "Talk Show", "Variety Show", "War", "Young Adult", "None"
# ]


# ##############################################################
# # Add Prediction Tools Section Below Existing Visualizations #
# ##############################################################
# st.markdown("---")  # Separator for clarity
# st.header("Predict Campaign Performance")
# st.markdown("Test the performance of a theoretical or previously untested campaign that wasn’t included in the training data using three insightful models. The first predicts the average watch time per device, the second forecasts the campaign score, and the third identifies the most similar campaign from the training data for comparison. For details on model accuracy and development, refer to the accompanying paper. The metric used here is average minutes watched per device.")

# # Shared Inputs for Prediction Tools
# impressions = st.number_input("Impressions", min_value=0)
# clicks = st.number_input("Clicks", min_value=0)
# genres = {genre: st.slider(genre, 0.0, 1.0) for genre in genre_columns}

# # Radio button to select Prediction Tool
# st.markdown("### Choose a Prediction Tool")
# option = st.radio(
#     "Select a tool:",
#     ["Predict Watch Time", "Predict Campaign Score", "Most Similar Campaign"]
# )

# if option == "Predict Watch Time":
#     if st.button("Predict Watch Time"):
#         user_input = np.array([impressions, clicks] + list(genres.values())).reshape(1, -1)
#         prediction = reg_model.predict(user_input)
#         st.write(f"Predicted Average Watch Time per Device: {prediction[0]:.2f}")

# elif option == "Predict Campaign Score":
#     if st.button("Predict Campaign Score"):
#         user_input = np.array([impressions, clicks] + list(genres.values())).reshape(1, -1)
#         score_prediction = rf_classifier.predict(user_input)
#         score_mapping = {0: 'Poor', 1: 'Fair', 2: 'Good', 3: 'Excellent'}
#         st.write(f"Predicted Campaign Score: {score_mapping[score_prediction[0]]}")

# elif option == "Most Similar Campaign":
#     if st.button("Find Similar Campaign"):
#         user_input = np.array([impressions, clicks] + list(genres.values())).reshape(1, -1)
#         user_input_scaled = scaler.transform(user_input)

#         # Find nearest campaign
#         distances, indices = nn_model.kneighbors(user_input_scaled)
#         similar_campaign = campaign_data.iloc[indices[0][0]]

#         st.write("### Most Similar Campaign Details:")
#         st.write(similar_campaign)

#         # Downloadable CSV of the similar campaign
#         csv = similar_campaign.to_frame().transpose().to_csv(index=False)
#         st.download_button(
#             label="Download Similar Campaign as CSV",
#             data=csv,
#             file_name='similar_campaign.csv',
#             mime='text/csv'
#         )
