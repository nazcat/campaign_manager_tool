import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
from google.cloud import storage
from google.oauth2 import service_account


#################################
# Set dashboard page parameters #
#################################
# Set page parameters
st.set_page_config(layout='wide', initial_sidebar_state='expanded')


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

# Set up Google Cloud credentials using secrets stored in Streamlit Cloud
service_key = {
  "type": "service_account",
  "project_id": st.secrets['PROJECT_ID'],
  "private_key_id": st.secrets['PRIVATE_KEY_ID'],
  "private_key": st.secrets['PRIVATE_KEY'],
  "client_email": st.secrets['CLIENT_EMAIL'],
  "client_id": st.secrets['CLIENT_ID'],
  "auth_uri": st.secrets['AUTH_URI'],
  "token_uri": st.secrets['TOKEN_URI'],
  "auth_provider_x509_cert_url": st.secrets['AUTH_PROVIDER_URL'],
  "client_x509_cert_url": st.secrets['CLIENT_CERT_URL'],
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


#####################################
# Streamlit Setup and Sidebar Start #
#####################################
def campaign_performance():
    st.markdown(f"# {list(page_names_to_funcs.keys())[0]}")

    # add date filter
    st.sidebar.markdown('## Streaming Dates')
    start_date = st.sidebar.date_input("Start Date", value=min(pd.to_datetime(totals['event_date'])))
    end_date = st.sidebar.date_input("End Date", value=max(pd.to_datetime(totals['event_date'])))                               

    # add multi select state filter
    st.sidebar.markdown('## Choose Campaign')
    all_campaigns = totals['campaign_name'].unique()
    select_all_campaigns = st.sidebar.checkbox("Select All Campaigns", value=False)

    # add filter logic if all campaigns are selected
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

    # add filter logic if all states are selected
    if select_all_states:
        state_select = all_states
    else:
        state_select = st.sidebar.multiselect(
            "Select State(s)", 
            options=all_states, 
            default=["MI"] if "MI" in all_states else []
        )

    # add multi select metric filter
    st.sidebar.markdown('## Choose Metric')
    metric_options = ['Users', 'Minutes', 'Impressions', 'Clicks', 'Minutes per User']
    metric_select = st.sidebar.selectbox("Select Metric", options=metric_options)


    #######################################
    # [Top Level] Totals by Date Selection #
    #######################################
    # Filter data for selected dates, campaigns, and states
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
        st.subheader('Totals')
        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Users", devices)
        col2.metric("Minutes Watched", minutes)
        col3.metric("Ad Impressions", impressions)
        col4.metric("Ad Clicks", clicks)
        col5.metric("Minutes per User", mpu)


    ###############################
    # [Visual #1] Totals by State #
    ###############################
    # Filter data for selected dates, campaigns, and states
    filtered_map_df = totals[(pd.to_datetime(totals['event_date']) >= pd.to_datetime(start_date)) & (pd.to_datetime(totals['event_date']) <= pd.to_datetime(end_date)) & (totals['state'].isin(state_select)) & (totals['campaign_name'].isin(campaign_select))].copy()

    filtered_map_agg = filtered_map_df.groupby(['state','latitude','longitude']).agg(
        {'Users': np.sum, 
         'Minutes': np.sum,
         'Impressions': np.sum,
         'Clicks': np.sum,
         'Minutes per User': np.mean
         }).round(1).reset_index()

    # scatter geo version
    fig1 = px.scatter_geo(
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
    fig1.update_layout(coloraxis_colorbar={"title": ""})

    # Remove color bar
    fig1.update_layout(coloraxis_showscale=False)

    # Display the chart and aggregated metric
    # metric_sum = round(filtered_map_agg[metric_select],1)
    # st.sidebar.metric(label=f"Total {metric_select.replace('_', ' ').title()} by Selected States(s)", value=filtered_map_agg[metric_select])


    ##########################################
    # [Visual #2] Users by Marketing Partner #
    ##########################################
    # Filter data for selected dates, campaigns, and states
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
    fig2 = px.pie(
        filtered_partner_agg,
        names='marketing_partner',    
        values=metric_select,            
        # title='Users by Marketing Partner',
        hover_data={metric_select: True, 'percentage': True},  # Hover details
        # labels={f"{metric_select}, 'percentage': 'Percentage'"},
        color_discrete_sequence=color_theme  # Apply the color theme
    )

    # Customize hover template to display total device_id and percentage
    fig2.update_traces(
        textinfo='percent',            # Show percentages on the pie chart
        hovertemplate='<b>%{label}</b><br>Total: %{value}<br>Percentage: %{percent}'
    )


    ###############################
    # [Visual #3] Totals by Genre #
    ###############################
    # Filter data for selected dates, campaigns, and states
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
    fig3 = px.bar(
        filtered_genre_agg,
        x='Genre',
        y=metric_select,
        # title="Number of Users by Genre",
        labels={f"{metric_select}, 'Genre'"}
    )

    fig3.update_layout(
        width=800, 
        height=400,
        xaxis=dict(
            tickangle=-45  # Rotate x-axis labels by -45 degrees
            )
    )


    ###############################################
    # [Visual #4] Days from Campaign to Streaming #
    ###############################################
    # Filter data for selected dates, campaigns, and states
    filtered_date_diff_df = totals[(pd.to_datetime(totals['event_date']) >= pd.to_datetime(start_date)) & (pd.to_datetime(totals['event_date']) <= pd.to_datetime(end_date)) & (totals['state'].isin(state_select)) & (totals['campaign_name'].isin(campaign_select))]

    if metric_select == 'minutes_per_user':
        filtered_date_diff_agg = filtered_date_diff_df.groupby(['imp_evnt_binned']).agg({metric_select: np.mean}).reset_index()
    else:
        filtered_date_diff_agg = filtered_date_diff_df.groupby(['imp_evnt_binned']).agg({metric_select: np.sum}).reset_index()

    fig4 = plt.figure(figsize=(6, 6))

    bars = plt.bar(filtered_date_diff_agg['imp_evnt_binned'], filtered_date_diff_agg[metric_select]) # '#00274C','#FFCB05'
    for bar in bars:
        yval = bar.get_height()
        plt.annotate(f'{int(yval/1000)}K',
                     xy=(bar.get_x() + bar.get_width() / 2, yval),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords='offset points',
                     ha='center', va='bottom')

    avg = filtered_date_diff_agg[metric_select].mean()

    # plot average
    plt.axhline(y=avg, ls='--', color='#FFCB05', label='Average') # '#00274C','#FFCB05'

    # annotate average line
    plt.annotate(f'Avg: {avg:.1f}', xy=(bar.get_x(), avg), xytext=(bar.get_x() + 1, avg + 50))

    # format plot
    plt.box(False)
    plt.ylabel(f"Total {metric_select}",fontsize=10)
    plt.xlabel('Days', fontsize=10)

    # Set the y limits making the maximum 10% greater
    ymin, ymax = min(filtered_date_diff_agg[metric_select]), max(filtered_date_diff_agg[metric_select])
    plt.ylim(ymin, 1.1 * ymax)
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
    # plt.title("Days from Campaign to Streaming")

    # Remove x and y tick lines
    plt.tick_params(axis='both', which='both', length=0)


    ####################################
    # [Visual #5] Campaign Time Series #
    ####################################
    # Filter data for selected dates, campaigns, and states
    filtered_trend_df = totals[(pd.to_datetime(totals['event_date']) >= pd.to_datetime(start_date)) & (pd.to_datetime(totals['event_date']) <= pd.to_datetime(end_date)) & (totals['state'].isin(state_select)) & (totals['campaign_name'].isin(campaign_select))].copy()

    if metric_select == 'minutes_per_user':
        filtered_trend_agg = filtered_trend_df.groupby(['event_date']).agg({metric_select: np.mean}).round(1).reset_index()
    else:
        filtered_trend_agg = filtered_trend_df.groupby(['event_date']).agg({metric_select: np.sum}).round(1).reset_index()

    fig5, ax = plt.subplots(figsize=(6, 5))

    #forecasts
    plt.plot(pd.to_datetime(filtered_trend_agg['event_date']), filtered_trend_agg[metric_select])

    # format plot
    plt.box(False)
    plt.ylabel(f"Total {metric_select}",fontsize=10)
    plt.xlabel('Watch Date', fontsize=10)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1, byweekday=mdates.MO))  # Major ticks every Monday
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) 

    # Remove x and y tick lines
    plt.tick_params(axis='both', which='both', length=0)
    plt.xticks(rotation=45)


    ############################
    # Deploy Streamlit Visuals #
    ############################
     # Row 1
    col1, col2 = st.columns((1,1))
    with col1:
        st.markdown(f"##### {metric_select.replace('_', ' ').title()} by State")
        st.plotly_chart(fig1)
    with col2:
        st.markdown(f'##### {metric_select} by Marketing Partner')
        st.plotly_chart(fig2)
        
    # # Row 2
    # st.markdown('### Line chart')
    col1, col2, col3 = st.columns((1,1,1))
    with col1:
        st.markdown(f"##### {metric_select} by Genre")
        st.markdown("###### For Genres >= 10 Users")
        st.plotly_chart(fig3)
    with col2:
        st.markdown("##### Days from Campaign to Streaming")
        st.pyplot(fig4)
    with col3:
        st.markdown(f"##### {metric_select} Over Time")
        st.pyplot(fig5)

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


    #########################
    # Streamlit Sidebar End #
    #########################
    st.sidebar.markdown('''
    ---
    Created with ❤️ by Plutonians.
    ''')


##############################################################
# Add Prediction Tools Section Below Existing Visualizations #
##############################################################
def campaign_predictions():
    st.markdown(f"# {list(page_names_to_funcs.keys())[1]}")
    st.markdown("---")  # Separator for clarity
    st.subheader("Predict Campaign Performance")
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


#################
# Combine pages #
#################
page_names_to_funcs = {
    "Campaign Performance": campaign_performance,
    "Campaign Planner": campaign_predictions,
}

campaign_manager = st.sidebar.selectbox("💡 Select Campaign Performance or Planner", page_names_to_funcs.keys())
page_names_to_funcs[campaign_manager]()
