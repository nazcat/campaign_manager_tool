import streamlit as st
import pandas as pd # version 2.0.3
import numpy as np
import plotly.express as px # version 5.15.0
import matplotlib.pyplot as plt # version 3.7.1
import matplotlib.ticker as mtick
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.dates as mdates


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


#####################################
# Streamlit Setup and Sidebar Start #
#####################################
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# add date filter
st.sidebar.header('')
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

# Filter data for selected dates, campaigns, and states
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


##############################################################
# [Visual #1] Total Users or Average Minutes Watched by State #
##############################################################
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

df_imp_evnt_agg = filtered_date_diff_df.groupby('imp_evnt_binned').agg(
    {'Users': np.sum, 
     'Minutes': np.sum,
     'Impressions': np.sum,
     'Clicks': np.sum,
     'Minutes per User': np.mean
     }).round(1).reset_index()

fig4 = plt.figure(figsize=(6, 6))

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


####################################
# [Visual #5] Campaign Time Series #
####################################
# Filter data for selected dates, campaigns, and states
filtered_trend_df = totals[(pd.to_datetime(totals['event_date']) >= pd.to_datetime(start_date)) & (pd.to_datetime(totals['event_date']) <= pd.to_datetime(end_date)) & (totals['state'].isin(state_select)) & (totals['campaign_name'].isin(campaign_select))].copy()

if metric_select == 'minutes_per_user':
    trend_df = filtered_trend_df.groupby(['event_date']).agg({metric_select: np.mean}).reset_index()
else:
    trend_df = filtered_trend_df.groupby(['event_date']).agg({metric_select: np.sum}).reset_index()

fig5, ax = plt.subplots(figsize=(6, 5))

#forecasts
plt.plot(pd.to_datetime(trend_df['event_date']), trend_df[metric_select])

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
