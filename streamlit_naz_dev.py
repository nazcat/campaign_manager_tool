import streamlit as st
import pandas as pd # version 2.0.3
import numpy as np
import plotly.express as px # version 5.15.0
import matplotlib.pyplot as plt # version 3.7.1
import matplotlib.ticker as mtick


############################
# load files for dashboard #
############################
totals = pd.read_csv('data/totals.csv')
totals_genre = pd.read_csv('data/totals_genre.csv')


#####################################
# Streamlit Setup and Sidebar Start #
#####################################
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# with open('style.css') as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
st.sidebar.header('Campaign Performance')

# add date filter
st.sidebar.subheader('Streaming Dates')
start_date = st.sidebar.date_input("Start Date", value=min(pd.to_datetime(totals['event_date'])))
end_date = st.sidebar.date_input("End Date", value=max(pd.to_datetime(totals['event_date'])))                               

# add multi select state filter
st.sidebar.subheader('Campaign')
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
st.sidebar.subheader('Choose State')
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

metric_options = ['users', 'minutes', 'impressions', 'clicks', 'minutes_per_user']
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
    devices = format_number(filtered_df['users'].sum())
    minutes = format_number(filtered_df['minutes'].sum())
    impressions = format_number(filtered_df['impressions'].sum())
    clicks = format_number(filtered_df['clicks'].sum())
    mpu = format_number(filtered_df['minutes_per_user'].mean())
    
    # update dashboard
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

    labels = filtered_partner_df['marketing_partner']
    sizes = filtered_partner_df['users']

    total_device_id = filtered_partner_df['users'].sum()
    filtered_partner_df['percentage'] = filtered_partner_df['users'] / total_device_id * 100

    # define color theme for chart
    color_theme = px.colors.qualitative.Set3

    # Create the Pie Chart
    fig1 = px.pie(
        filtered_partner_df,
        names='marketing_partner',      # Labels
        values='users',             # Values
        # title='Users by Marketing Partner',
        hover_data={'users': True, 'percentage': True},  # Hover details
        labels={'users': 'Total Users', 'percentage': 'Percentage'},
        color_discrete_sequence=color_theme  # Apply the color theme
    )

    # Customize hover template to display total device_id and percentage
    fig1.update_traces(
        textinfo='percent',            # Show percentages on the pie chart
        hovertemplate='<b>%{label}</b><br>Total Devices: %{value}<br>Percentage: %{percent}'
    )


##############################################
# [Visual 2] Days from Campaign to Streaming #
##############################################
# Apply date filters from sidebar selection
if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")

else:
    # Filter DataFrame by selected dates
    filtered_date_diff_df = totals[(pd.to_datetime(totals['event_date']) >= pd.to_datetime(start_date)) & (pd.to_datetime(totals['event_date']) <= pd.to_datetime(end_date)) & (totals['state'].isin(state_select)) & (totals['campaign_name'].isin(campaign_select))]

    df_imp_evnt_agg = filtered_date_diff_df.groupby('imp_evnt_binned')['users'].sum().reset_index()

    fig2 = plt.figure(figsize=(10, 6))

    bars = plt.bar(df_imp_evnt_agg['imp_evnt_binned'], df_imp_evnt_agg['users']) # '#00274C','#FFCB05'
    for bar in bars:
        yval = bar.get_height()
        plt.annotate(f'{int(yval/1000)}K',
                     xy=(bar.get_x() + bar.get_width() / 2, yval),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords='offset points',
                     ha='center', va='bottom')

    # plot average
    plt.axhline(y=df_imp_evnt_agg['users'].mean(), ls='--', label='Average') # '#00274C','#FFCB05'

    # format plot
    plt.box(False)
    plt.ylabel('Total Users',fontsize=10)
    plt.xlabel('Days', fontsize=10)

    # Set the y limits making the maximum 10% greater
    ymin, ymax = min(df_imp_evnt_agg['users']), max(df_imp_evnt_agg['users'])
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

    # Calculate aggregated metric for selected states
    if metric_select == 'minutes_per_user':
        metric_sum = round(filtered_map_df[metric_select].mean(),0)
    else:
        metric_sum = round(filtered_map_df[metric_select].sum(),0)

    # Choropleth version
    # fig3 = px.choropleth(
    #     filtered_map_df,
    #     locations="state",
    #     locationmode="USA-states",
    #     color=metric_select,
    #     hover_name="state",
    #     hover_data={metric_select: True},
    #     color_continuous_scale="Viridis",
    #     scope="usa",
    #     title=f"{metric_select.replace('_', ' ').title()} by State"
    # )
    
    # scatter geo version
    fig3 = px.scatter_geo(
        filtered_map_df,
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
    st.sidebar.metric(label=f"Total {metric_select.replace('_', ' ').title()} by Selected States(s)", value=metric_sum)


##############################
# [Visual 4] Totals by Genre #
##############################
# Filter data based on user selections
if select_all_states:
    filtered_genre_df = totals_genre.copy()

else:
    filtered_genre_df = totals_genre[(pd.to_datetime(totals_genre['event_date']) >= pd.to_datetime(start_date)) & (pd.to_datetime(totals_genre['event_date']) <= pd.to_datetime(end_date)) & (totals_genre['state'].isin(state_select)) & (totals_genre['campaign_name'].isin(campaign_select))].copy()

    filtered_genre_agg = filtered_genre_df.groupby(['content_genre']).agg(
        {'users': pd.Series.nunique, 
         'minutes': np.sum,
         'impressions': np.sum,
         'clicks': np.sum,
         'minutes_per_user': np.mean}).reset_index()

    # Bar Chart
    fig4 = px.bar(
        filtered_genre_agg,
        x='content_genre',
        y='users',
        # title="Number of Users by Genre",
        labels={'users': 'Users', 'content_genre': 'Genre'}
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
    st.markdown('##### Users by Marketing Partner')
    st.plotly_chart(fig1)
    
# # Row 2
# st.markdown('### Line chart')
col1, col2 = st.columns((1,1))
with col1:
    st.markdown("##### Number of Users by Genre")
    st.markdown("###### For Genres >= 10 Users")
    st.plotly_chart(fig4)
with col2:
    st.markdown("##### Days from Campaign to Streaming")
    st.pyplot(fig2)

# # Row 3
# Add downloadable campaign crosstab
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
