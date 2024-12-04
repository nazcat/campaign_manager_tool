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
# df_imp_evnt_agg = pd.read_csv('data/install_event_binned_users.csv')
date_diff_totals = pd.read_csv('data/install_event_binned_date.csv')
partners = pd.read_csv('data/marketing_partners_users.csv')
states = pd.read_csv('data/states_totals.csv')


#####################################
# Streamlit Setup and Sidebar Start #
#####################################
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# with open('style.css') as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
st.sidebar.header('`Campaign Performance`')

# add multi select state filter
st.sidebar.subheader('Choose State')
state_select = st.sidebar.multiselect("Select State", ["All"] + states['state'].tolist())

# add date filter
st.sidebar.subheader('Streaming Dates')
start_date = st.sidebar.date_input("Start Date", value=min(pd.to_datetime(totals['event_date'])))
end_date = st.sidebar.date_input("End Date", value=max(pd.to_datetime(totals['event_date'])))                               


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
    
    return f'{num // 1000} K'

# Apply date filters from sidebar selection
if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")

else:
    # Filter DataFrame by selected dates
    filtered_df = totals[(pd.to_datetime(totals['event_date']) >= pd.to_datetime(start_date)) & (pd.to_datetime(totals['event_date']) <= pd.to_datetime(end_date))]

    # top four metrics
    devices = format_number(filtered_df['users'].sum())
    minutes = format_number(filtered_df['minutes'].sum())
    impressions = format_number(filtered_df['impressions'].sum())
    clicks = format_number(filtered_df['clicks'].sum())
    mpu = format_number(filtered_df['minutes_per_user'].mean())
    
    # update dashboard
    st.markdown('### Totals')
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Users", devices) #, "1.2 °F")
    col2.metric("Minutes Watched", minutes) #, "-8%")
    col3.metric("Ad Impressions", impressions) #, "4%")
    col4.metric("Ad Clicks", clicks) # "4%")
    col5.metric("Minutes per User", mpu) # "4%")


#########################################
# [Visual 1] Users by Marketing Partner #
#########################################
if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")

else:
    # Filter DataFrame by selected dates
    filtered_partner_df = partners[(pd.to_datetime(partners['event_date']) >= pd.to_datetime(start_date)) & (pd.to_datetime(partners['event_date']) <= pd.to_datetime(end_date))]

    labels = filtered_partner_df['marketing_partner']
    sizes = filtered_partner_df['device_id']

    total_device_id = filtered_partner_df['device_id'].sum()
    filtered_partner_df['percentage'] = filtered_partner_df['device_id'] / total_device_id * 100

    # define color theme for chart
    color_theme = px.colors.qualitative.Set3

    # Create the Pie Chart
    fig1 = px.pie(
        filtered_partner_df,
        names='marketing_partner',      # Labels
        values='device_id',             # Values
        title='Users by Marketing Partner',
        hover_data={'device_id': True, 'percentage': True},  # Hover details
        labels={'device_id': 'Total Users', 'percentage': 'Percentage'},
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
if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")

else:
    # Filter DataFrame by selected dates
    filtered_date_diff_df = date_diff_totals[(pd.to_datetime(date_diff_totals['event_date']) >= pd.to_datetime(start_date)) & (pd.to_datetime(date_diff_totals['event_date']) <= pd.to_datetime(end_date))]

    # bin impression to event date diffs
    imp_inst_bins = [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 120]
    imp_inst_labels = ['1','2','3','4','5','6','7','8','9','10','11-120']
    filtered_date_diff_df['imp_evnt_binned'] = pd.cut(filtered_date_diff_df['impression_event_date_diff'], bins=imp_inst_bins, labels=imp_inst_labels)
    del filtered_date_diff_df['impression_event_date_diff']

    df_imp_evnt_agg = filtered_date_diff_df.groupby('imp_evnt_binned')['device_id'].sum().reset_index()

    fig2 = plt.figure(figsize=(6, 4))

    bars = plt.bar(df_imp_evnt_agg['imp_evnt_binned'], df_imp_evnt_agg['device_id'], color='#00274C') # '#00274C','#FFCB05'
    for bar in bars:
        yval = bar.get_height()
        plt.annotate(f'{int(yval/1000)}K',
                     xy=(bar.get_x() + bar.get_width() / 2, yval),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords='offset points',
                     ha='center', va='bottom')

    # plot average
    plt.axhline(y=df_imp_evnt_agg['device_id'].mean(), color='#FFCB05', ls='--', label='Average') # '#00274C','#FFCB05'

    plt.ylabel('Total Users',fontsize=10)
    plt.xlabel('Days', fontsize=10)

    # Set the y limits making the maximum 10% greater
    ymin, ymax = min(df_imp_evnt_agg['device_id']), max(df_imp_evnt_agg['device_id'])
    plt.ylim(ymin, 1.1 * ymax)
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
    plt.title("Days from Campaign to Streaming")
    plt.xticks(rotation=45)


##############################################################
# [Visual 3] Total Users or Average Minutes Watched by State #
##############################################################
# add multi select state filter
st.sidebar.subheader('Choose State')
state_select = st.sidebar.multiselect("Select States", options=states['state'].tolist(), default=states['state'].tolist())

st.sidebar.subheader('Choose Metric')
metric_options = ['users', 'minutes', 'impressions', 'clicks', 'minutes_per_user']
selected_metric = st.sidebar.selectbox("Select Metric", options=metric_options)

# # Initialize figure with the first metric
# metric_options = ['users', 'minutes', 'impressions', 'clicks', 'minutes_per_user']
# initial_metric = metric_options[0]

# Filter data for selected states
if state_select:
    filtered_df = states[states['state'].isin(state_select)].copy()
    filtered_df['opacity'] = 1  # Selected states full opacity

else:
    filtered_df = states.copy()
    filtered_df['opacity'] = 1  # No states selected, all visible

# Calculate aggregated metric for selected states
metric_sum = filtered_df[selected_metric].sum()

# Add greyed-out opacity for unselected states
states['opacity'] = states['state'].apply(lambda x: 1 if x in state_select else 0.2)

# Create the Choropleth Map
fig3 = px.choropleth(
    filtered_df,
    locations="state",
    locationmode="USA-states",
    color=selected_metric,
    hover_name="state",
    hover_data={selected_metric: True},
    color_continuous_scale="Viridis",
    scope="usa",
    title=f"{selected_metric.replace('_', ' ').title()} by State",
)

# Adjust opacity for greyed-out states
fig3.update_traces(marker_opacity=states['opacity'])

# Update layout (remove colorbar title)
fig3.update_layout(coloraxis_colorbar={"title": ""})

# Display the chart and aggregated metric
st.sidebar.metric(label=f"Total {selected_metric.replace('_', ' ').title()} by Selected State(s)", value=metric_sum)


############################
# Deploy Streamlit Visuals #
############################
# Row 1
col1, col2 = st.columns((6,4))
with col1:
    st.plotly_chart(fig3)
with col2:
    st.plotly_chart(fig1)

# # Row 2
# st.markdown('### Line chart')
col1, col2, col3 = st.columns((3,3,3))
with col1:
    st.pyplot(fig2)


#########################
# Streamlit Sidebar End #
#########################
st.sidebar.markdown('''
---
Created with ❤️ by Plutonians.
''')
