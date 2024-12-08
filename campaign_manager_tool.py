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

############################
# load files for dashboard #
############################
totals = pd.read_csv('data/totals.csv')
totals_genre = pd.read_csv('data/totals_genre.csv')

# rename columns for visuals
totals = totals.rename(
    columns={
    'users': 'Users',
    'minutes': 'Minutes',
    'impressions': 'Impressions',
    'clicks': 'Clicks',
    'minutes_per_user': 'Minutes per User'
    })

# rename columns for visuals
totals_genre = totals_genre.rename(
    columns={
    'users': 'Users',
    'minutes': 'Minutes',
    'impressions': 'Impressions',
    'clicks': 'Clicks',
    'minutes_per_user': 'Minutes per User'
    })

##########################################
# load file for models from Google Cloud #
##########################################
# Load file anon_processed_unique_device_v3.csv from GCP


# For GitHub Actions, you access secrets as environment variables
my_secret_key = os.getenv("MY_SECRET_KEY")

# Use the secret value in your app

# Set up Google Cloud credentials
service_key = {
  "type": "service_account",
  "project_id": os.getenv('PROJECT_ID'),
  "private_key_id": os.getenv('PRIVATE_KEY_ID'),
  "private_key": os.getenv('PRIVATE_KEY'),
  "client_email": os.getenv('CLIENT_EMAIL'),
  "client_id": os.getenv('CLIENT_ID'),
  "auth_uri": os.getenv('AUTH_URI'),
  "token_uri": os.getenv('TOKEN_URI'),
  "auth_provider_x509_cert_url": os.getenv('AUTH_PROVIDER_URL'),
  "client_x509_cert_url": os.getenv('CLIENT_CERT_URL'),
  "universe_domain": "googleapis.com"
}

#Downloaded credentials in JSON format
project_id=service_key["project_id"]
credentials = service_account.Credentials.from_service_account_info(service_key)
client = storage.Client(project=project_id,credentials=credentials)

# Access the file in the bucket
bucket_name = 'campaign_manager_tool'
file_name = 'anon_processed_unique_device_v3.csv'
bucket = client.bucket(bucket_name)
blob = bucket.blob(file_name)

# Download the file content and read as a dataframe
content = blob.download_as_bytes()
anon_df = pd.read_csv(io.BytesIO(content))


############################
# Load Datasets for Models #
############################
@st.cache_data
def load_data():
    engagement_data = anon_df
    campaign_data = pd.read_csv('data/anan_campaign_modeling_data_v3.csv')
    return engagement_data, campaign_data

# Load Models for Prediction Tools
@st.cache_resource
def load_models():
    reg_model = joblib.load(r'data/models/regression_model.pkl')
    rf_classifier = joblib.load(r'data/models/rf_classifier.pkl')
    nn_model = joblib.load(r'data/models/nearest_neighbors_model.pkl')
    scaler = joblib.load(r'data/models/nearest_neighbors_scaler.pkl')
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

#####################################
# Streamlit Setup and Sidebar Start #
#####################################
#st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# with open('style.css') as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.sidebar.header('')

# add date filter
st.sidebar.markdown('## Streaming Dates')
start_date = st.sidebar.date_input("Start Date", value=min(pd.to_datetime(totals['event_date'])))
end_date = st.sidebar.date_input("End Date", value=max(pd.to_datetime(totals['event_date'])))                               

# add multi select state filter
st.sidebar.markdown('## Choose Campaign')
all_campaigns = totals['campaign_name'].unique()
select_all_campaigns = st.sidebar.checkbox("Select All Campaigns", value=False)

if select_all_campaigns:
    campaign_select = all_campaigns
else:
    campaign_select = st.sidebar.multiselect(
        "Select State(s)", 
        options=all_campaigns, 
        default=["campaign_name_1"] if "campaign_name_1" in all_campaigns else []
    )
    
# add multi select state filter
st.sidebar.markdown('## Choose State')
all_states = totals['state'].unique()
select_all_states = st.sidebar.checkbox("Select All States", value=False)

if select_all_states:
    state_select = all_states
else:
    state_select = st.sidebar.multiselect(
        "Select State(s)", 
        options=all_states, 
        default=["MI"] if "MI" in all_states else []
    )

metric_options = ['Users', 'Minutes', 'Impressions', 'Clicks', 'Minutes per User']
metric_select = st.sidebar.selectbox("Select Metric", options=metric_options)


#######################################
# [Top Leve] Totals by Date Selection #
#######################################
# number formatter
def format_number(num):
    if num > 1000000000:
        if not num % 1000000000:
            return f'{num // 1000000000} B'
        return f'{round(num / 1000000000, 1)} B'
    
    elif num > 1000000:
        if not num % 1000000:
            return f'{num // 1000000} M'
        return f'{round(num / 1000000, 1)} M'

    elif num > 1000:
        if not num % 1000:
            return f'{num // 1000} K'
        return f'{round(num / 1000, 1)} K'

    return f'{round(num,1)}'

# Apply date filters from sidebar selection
if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")

else:
    # Filter DataFrame by selected dates
    filtered_df = totals[(pd.to_datetime(totals['event_date']) >= pd.to_datetime(start_date)) & (pd.to_datetime(totals['event_date']) <= pd.to_datetime(end_date)) & (totals['state'].isin(state_select)) & (totals['campaign_name'].isin(campaign_select))]
 
    # top four metrics
    devices = format_number(filtered_df['Users'].sum())
    minutes = format_number(filtered_df['Minutes'].sum())
    impressions = format_number(filtered_df['Impressions'].sum())
    clicks = format_number(filtered_df['Clicks'].sum())
    mpu = format_number(filtered_df['Minutes per User'].mean())
    
    # update dashboard
    st.header('Campaign Performance')
    st.subheader('Totals')
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Users", devices)
    col2.metric("Minutes Watched", minutes)
    col3.metric("Ad Impressions", impressions)
    col4.metric("Ad Clicks", clicks)
    col5.metric("Minutes per User", mpu)


#########################################
# [Visual 1] Users by Marketing Partner #
#########################################
if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")

else:
    # Filter DataFrame by selected dates
    filtered_partner_df = totals[(pd.to_datetime(totals['event_date']) >= pd.to_datetime(start_date)) & (pd.to_datetime(totals['event_date']) <= pd.to_datetime(end_date)) & (totals['state'].isin(state_select)) & (totals['campaign_name'].isin(campaign_select))]

    filtered_partner_agg = filtered_partner_df.groupby(['marketing_partner']).agg(
        {'Users': np.sum, 
         'Minutes': np.sum,
         'Impressions': np.sum,
         'Clicks': np.sum,
         'Minutes per User': np.mean
         }).round(1).reset_index()

    labels = filtered_partner_agg['marketing_partner']
    sizes = filtered_partner_agg[metric_select]

    total_metric = filtered_partner_agg[metric_select].sum()
    filtered_partner_agg['percentage'] = round((filtered_partner_agg[metric_select] / total_metric * 100),1)

    # define color theme for chart
    color_theme = px.colors.qualitative.Set3

    # Create the Pie Chart
    fig1 = px.pie(
        filtered_partner_agg,
        names='marketing_partner',    
        values=metric_select,            
        # title='Users by Marketing Partner',
        hover_data={metric_select: True, 'percentage': True},  # Hover details
        # labels={f"{metric_select}, 'percentage': 'Percentage'"},
        color_discrete_sequence=color_theme  # Apply the color theme
    )

    # Customize hover template to display total device_id and percentage
    fig1.update_traces(
        textinfo='percent',            # Show percentages on the pie chart
        hovertemplate='<b>%{label}</b><br>Total: %{value}<br>Percentage: %{percent}'
    )


##############################################
# [Visual 2] Days from Campaign to Streaming #
##############################################
# Apply date filters from sidebar selection
if select_all_states:
    filtered_map_df = totals.copy()

else:
    # Filter DataFrame by selected dates
    filtered_date_diff_df = totals[(pd.to_datetime(totals['event_date']) >= pd.to_datetime(start_date)) & (pd.to_datetime(totals['event_date']) <= pd.to_datetime(end_date)) & (totals['state'].isin(state_select)) & (totals['campaign_name'].isin(campaign_select))]

    df_imp_evnt_agg = filtered_date_diff_df.groupby('imp_evnt_binned').agg(
        {'Users': np.sum, 
         'Minutes': np.sum,
         'Impressions': np.sum,
         'Clicks': np.sum,
         'Minutes per User': np.mean
         }).round(1).reset_index()
    
    fig2 = plt.figure(figsize=(10, 6))

    bars = plt.bar(df_imp_evnt_agg['imp_evnt_binned'], df_imp_evnt_agg[metric_select]) # '#00274C','#FFCB05'
    for bar in bars:
        yval = bar.get_height()
        plt.annotate(f'{int(yval/1000)}K',
                     xy=(bar.get_x() + bar.get_width() / 2, yval),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords='offset points',
                     ha='center', va='bottom')

    avg = df_imp_evnt_agg[metric_select].mean()
    
    # plot average
    plt.axhline(y=avg, ls='--', color='#FFCB05', label='Average') # '#00274C','#FFCB05'

    # annotate average line
    plt.annotate(f'Avg: {avg:.1f}', xy=(bar.get_x(), avg), xytext=(bar.get_x() + 1, avg + 50))

    # format plot
    plt.box(False)
    plt.ylabel(f"Total {metric_select}",fontsize=10)
    plt.xlabel('Days', fontsize=10)

    # Set the y limits making the maximum 10% greater
    ymin, ymax = min(df_imp_evnt_agg[metric_select]), max(df_imp_evnt_agg[metric_select])
    plt.ylim(ymin, 1.1 * ymax)
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
    # plt.title("Days from Campaign to Streaming")

    # Remove x and y tick lines
    plt.tick_params(axis='both', which='both', length=0)


##############################################################
# [Visual 3] Total Users or Average Minutes Watched by State #
##############################################################
# Filter data for selected states
if select_all_states:
    filtered_map_df = totals.copy()

else:
    filtered_map_df = totals[(pd.to_datetime(totals['event_date']) >= pd.to_datetime(start_date)) & (pd.to_datetime(totals['event_date']) <= pd.to_datetime(end_date)) & (totals['state'].isin(state_select)) & (totals['campaign_name'].isin(campaign_select))].copy()

    filtered_map_agg = filtered_map_df.groupby(['state','latitude','longitude']).agg(
        {'Users': np.sum, 
         'Minutes': np.sum,
         'Impressions': np.sum,
         'Clicks': np.sum,
         'Minutes per User': np.mean
         }).round(1).reset_index()

    # scatter geo version
    fig3 = px.scatter_geo(
        filtered_map_agg,
        lat="latitude",
        lon="longitude",
        size=metric_select,
        color=metric_select,
        hover_name="state",
        hover_data={metric_select: True},
        # title=f"{metric_select.replace('_', ' ').title()} by State"
    )

    # Update layout (remove colorbar title)
    fig3.update_layout(coloraxis_colorbar={"title": ""})

    # Remove color bar
    fig3.update_layout(coloraxis_showscale=False)

    # Display the chart and aggregated metric
    # metric_sum = round(filtered_map_agg[metric_select],1)
    # st.sidebar.metric(label=f"Total {metric_select.replace('_', ' ').title()} by Selected States(s)", value=filtered_map_agg[metric_select])


##############################
# [Visual 4] Totals by Genre #
##############################
# Filter data based on user selections
if select_all_states:
    filtered_genre_df = totals_genre.copy()

else:
    filtered_genre_df = totals_genre[(pd.to_datetime(totals_genre['event_date']) >= pd.to_datetime(start_date)) & (pd.to_datetime(totals_genre['event_date']) <= pd.to_datetime(end_date)) & (totals_genre['state'].isin(state_select)) & (totals_genre['campaign_name'].isin(campaign_select))].copy()

    filtered_genre_agg = filtered_genre_df.groupby(['content_genre']).agg(
        {'Users': np.sum, 
         'Minutes': np.sum,
         'Impressions': np.sum,
         'Clicks': np.sum,
         'Minutes per User': np.mean
         }).round(1).reset_index()

    # rename columns for visuals
    filtered_genre_agg = filtered_genre_agg.rename(columns={'content_genre':'Genre'})
    
    # Bar Chart
    fig4 = px.bar(
        filtered_genre_agg,
        x='Genre',
        y=metric_select,
        # title="Number of Users by Genre",
        labels={f"{metric_select}, 'Genre'"}
    )

    fig4.update_layout(
        width=800, 
        height=400,
        xaxis=dict(
            tickangle=-45  # Rotate x-axis labels by -45 degrees
            )
    )


############################
# Deploy Streamlit Visuals #
############################
# Row 1
col1, col2 = st.columns((1,1))
with col1:
    st.markdown(f"##### {metric_select.replace('_', ' ').title()} by State")
    st.plotly_chart(fig3)
with col2:
    st.markdown(f'##### {metric_select} by Marketing Partner')
    st.plotly_chart(fig1)
    
# # Row 2
# st.markdown('### Line chart')
col1, col2 = st.columns((1,1))
with col1:
    st.markdown(f"##### {metric_select} by Genre")
    st.markdown("###### For Genres >= 10 Users")
    st.plotly_chart(fig4)
with col2:
    st.markdown("##### Days from Campaign to Streaming")
    st.pyplot(fig2)

# # Row 3
# Add downloadable campaign crosstab
# rename columns for visuals
filtered_genre_df = filtered_genre_df.rename(
    columns={
    'event_date': 'Watch Date',
    'campaign_name': 'Campaign',
    'content_genre': 'Genre',
    'state': 'State',
    })

st.markdown("##### Genres by Campaign(s)")
st.markdown("###### For Genres >= 10 Users")
st.dataframe(filtered_genre_df)


# Add Prediction Tools Section Below Existing Visualizations
st.markdown("---")  # Separator for clarity
st.header("Predict Campaign Performance")
st.markdown("Test the performance of a theoretical or previously untested campaign that wasn’t included in the training data using three insightful models. The first predicts the average watch time per device, the second forecasts the campaign score, and the third identifies the most similar campaign from the training data for comparison. For details on model accuracy and development, refer to the accompanying paper. The metric used here is average minutes watched per device.")

# Shared Inputs for Prediction Tools
impressions = st.number_input("Impressions", min_value=0)
clicks = st.number_input("Clicks", min_value=0)
genres = {genre: st.slider(genre, 0.0, 1.0) for genre in genre_columns}

# Radio button to select Prediction Tool
st.markdown("### Choose a Prediction Tool")
option = st.radio(
    "Select a tool:",
    ["Predict Watch Time", "Predict Campaign Score", "Most Similar Campaign"]
)

if option == "Predict Watch Time":
    if st.button("Predict Watch Time"):
        user_input = np.array([impressions, clicks] + list(genres.values())).reshape(1, -1)
        prediction = reg_model.predict(user_input)
        st.write(f"Predicted Average Watch Time per Device: {prediction[0]:.2f}")

elif option == "Predict Campaign Score":
    if st.button("Predict Campaign Score"):
        user_input = np.array([impressions, clicks] + list(genres.values())).reshape(1, -1)
        score_prediction = rf_classifier.predict(user_input)
        score_mapping = {0: 'Poor', 1: 'Fair', 2: 'Good', 3: 'Excellent'}
        st.write(f"Predicted Campaign Score: {score_mapping[score_prediction[0]]}")

elif option == "Most Similar Campaign":
    if st.button("Find Similar Campaign"):
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

#########################
# Streamlit Sidebar End #
#########################
st.sidebar.markdown('''
---
Created with ❤️ by Plutonians.
''')
