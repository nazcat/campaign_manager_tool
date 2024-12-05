import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load Datasets and Models
@st.cache_data
def load_data():
    engagement_data = pd.read_csv(r'outputs/anon_processed_unique_device_v3.csv')
    campaign_data = pd.read_csv('outputs/anan_campaign_modeling_data_v3.csv')
    return engagement_data, campaign_data

@st.cache_resource
def load_models():
    reg_model = joblib.load(r'models/regression_model.pkl')
    rf_classifier = joblib.load(r'models/rf_classifier.pkl')
    nn_model = joblib.load(r'models/nearest_neighbors_model.pkl')
    scaler = joblib.load(r'models/nearest_neighbors_scaler.pkl')
    return reg_model, rf_classifier, nn_model, scaler

# Preprocess Engagement Data
@st.cache_data
def preprocess_engagement_data(data):
    data['event_date'] = pd.to_datetime(data['event_date'].str[12:22])
    data['install_date'] = pd.to_datetime(data['install_date'].str[12:22])
    return data

# Load Data and Models
engagement_data, campaign_data = load_data()
engagement_data = preprocess_engagement_data(engagement_data)
reg_model, rf_classifier, nn_model, scaler = load_models()

# Columns for Genre Input
genre_columns = [
    "Classics", "Comedy", "Other", "Reality", "News and Information",
    "Drama", "Action & Adventure", "Thriller", "Sci-Fi & Fantasy", "Horror", "Western",
    "Documentaries", "Sports", "Instructional & Educational", "Home & Lifestyle", "Romance",
    "Anime", "Musical", "Independent", "Entertainment", "Paranormal", "Music", "Gay & Lesbian",
    "Crime", "Food & Cooking", "Faith & Spirituality", "Game Show", "Dance", "Children & Family",
    "Telenovela", "Talk Show", "Variety Show", "War", "Young Adult", "None"
]

# App Navigation
st.title("Campaign Insights and Engagement Dashboard")
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Menu", [
    "Home", 
    "Interactive Engagement Dashboard", 
    "Predict Watch Time", 
    "Predict Campaign Score",
    "Most Similar Campaign"
])

if menu == "Home":
    st.write("Move Naz streamlit to here")

elif menu == "Interactive Engagement Dashboard":
    st.subheader("Interactive Device Engagement Dashboard")

    # Sidebar filters
    st.sidebar.header("Filters")
    all_states = engagement_data['state'].unique()
    select_all_states = st.sidebar.checkbox("Select All States", value=False)

    if select_all_states:
        selected_states = all_states
    else:
        selected_states = st.sidebar.multiselect(
            "Select State(s)", 
            options=all_states, 
            default=["MI"] if "MI" in all_states else []
        )

    campaign_id_options = ["All Campaign IDs"] + list(engagement_data['campaign_id'].unique())
    selected_campaign_id = st.sidebar.selectbox(
        "Select Campaign ID", 
        options=campaign_id_options, 
        index=campaign_id_options.index("All Campaign IDs")
    )

    # Filter data based on user selections
    filtered_data = engagement_data[engagement_data['state'].isin(selected_states)]

    if selected_campaign_id != "All Campaign IDs":
        filtered_data = filtered_data[filtered_data['campaign_id'] == selected_campaign_id]

    # Metrics
    st.metric("Total Clicks", filtered_data['clicks'].sum())
    st.metric("Total Minutes Watched", filtered_data['total_min_watched'].sum())
    st.metric("Number of Devices", len(filtered_data))

    # Bar Chart
    genre_data = filtered_data[genre_columns].astype(bool).sum().reset_index()
    genre_data.columns = ['Genre', 'Number of Devices']
    genre_chart = px.bar(
        genre_data,
        x='Genre',
        y='Number of Devices',
        title="Number of Devices by Genre",
        labels={'Number of Devices': 'Devices', 'Genre': 'Content Genre'}
    )
    st.plotly_chart(genre_chart)

    # Geographic Engagement Heatmap
    geo_map = px.scatter_geo(
        filtered_data,
        lat="latitude",
        lon="longitude",
        size="total_min_watched",
        color="state",
        hover_name="state",
        title="Engagement by Geography"
    )
    st.plotly_chart(geo_map)

    # Data Table
    st.subheader("Filtered Data Table")
    st.dataframe(filtered_data)

elif menu == "Predict Watch Time":
    st.subheader("Predict Average Watch Time")

    impressions = st.number_input("Impressions", min_value=0)
    clicks = st.number_input("Clicks", min_value=0)
    genres = {genre: st.slider(genre, 0.0, 1.0) for genre in genre_columns}

    if st.button("Predict"):
        user_input = np.array([impressions, clicks] + list(genres.values())).reshape(1, -1)
        prediction = reg_model.predict(user_input)
        st.write(f"Predicted Average Watch Time per Device: {prediction[0]:.2f}")



elif menu == "Predict Campaign Score":
    st.subheader("Predict Campaign Score")
    st.write("This section uses a Random Forest model to predict the campaign score. Here are some details about the model's performance:")
    st.write("- **Accuracy**: 56%")
    st.write("- **Classification Report**:")
    st.code("""
              precision    recall  f1-score   support

        Poor       0.25      0.33      0.29         3
        Fair       0.33      0.50      0.40         2
        Good       1.00      0.83      0.91         6
   Excellent       0.50      0.40      0.44         5

    accuracy                           0.56        16
   macro avg       0.52      0.52      0.51        16
weighted avg       0.62      0.56      0.58        16
    """)
    st.write("- **Confusion Matrix**:")
    st.code("""
    [[1 1 0 1]
     [1 1 0 0]
     [0 0 5 1]
     [2 1 0 2]]
    """)
    st.subheader("Predict Campaign Score")

    impressions = st.number_input("Impressions", min_value=0)
    clicks = st.number_input("Clicks", min_value=0)
    genres = {genre: st.slider(genre, 0.0, 1.0) for genre in genre_columns}

    if st.button("Predict Score"):
        user_input = np.array([impressions, clicks] + list(genres.values())).reshape(1, -1)
        score_prediction = rf_classifier.predict(user_input)
        score_mapping = {0: 'Poor', 1: 'Fair', 2: 'Good', 3: 'Excellent'}
        st.write(f"Predicted Campaign Score: {score_mapping[score_prediction[0]]}")
elif menu == "Most Similar Campaign":
    st.subheader("Find the Most Similar Campaign")

    # Input form
    impressions = st.number_input("Impressions", min_value=0)
    clicks = st.number_input("Clicks", min_value=0)
    genres = {genre: st.slider(genre, 0.0, 1.0) for genre in genre_columns}

    if st.button("Find Similar Campaign"):
        # Create user input array
        user_input = np.array([impressions, clicks] + list(genres.values())).reshape(1, -1)
        user_input_scaled = scaler.transform(user_input)

        # Find nearest campaign
        distances, indices = nn_model.kneighbors(user_input_scaled)
        similar_campaign = campaign_data.iloc[indices[0][0]]

        st.write("### Most Similar Campaign Details:")
        st.write(similar_campaign)

        # Downloadable CSV of the similar campaign
        csv = similar_campaign.to_frame().transpose().to_csv(index=False)
        st.download_button(
            label="Download Similar Campaign as CSV",
            data=csv,
            file_name='similar_campaign.csv',
            mime='text/csv'
        )
