import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
# Set the default style for matplotlib
# plt.style.use('bmh')  # Using 'default' instead of 'seaborn'
plt.style.use('seaborn-v0_8-darkgrid')

# ['Solarize_Light2', '_classic_test_patch', 
# '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 
# 'classic', 'dark_background', 'fast', 'fivethirtyeight', 
# 'ggplot', 'grayscale', 'seaborn-v0_8', 'seaborn-v0_8-bright', 
# 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 
# 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 
# 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 
# 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 
# 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10']

# Set page config
st.set_page_config(page_title="VeBetterDAO: X-apps User Insights", layout="centered")

# Load the dataset
df = pd.read_csv('dataframe.csv')

# Convert Timestamp and FirstSeen to datetime objects
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['FirstSeen'] = pd.to_datetime(df['FirstSeen'])
df['DayOfWeek'] = pd.Categorical(df['Timestamp'].dt.day_name(), categories=[
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered=True)
df['HourOfDay'] = df['Timestamp'].dt.hour
df['DaysSinceFirst'] = (df['Timestamp'] - df['FirstSeen']).dt.total_seconds() / (60 * 60 * 24)  # in days
df['Weekend'] = df['Timestamp'].dt.dayofweek >= 5  # Weekend if True

st.title("VeBetterDAO: X-apps User Insights Dashboard")

st.write("This data primarily focuses on apps utilizing the Rewards Pool for B3TR distribution. This data only includes information up to October 04, 2024")



# Set the style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("whitegrid")
sns.set_palette("pastel")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['lines.linewidth'] = 2


chartSize = (12, 8)


# Create a function to display the visualizations
def create_visualizations():
 # NEW: Cumulative Total Actions Over Time
    st.subheader("Cumulative Total Actions Over Time")
    df_total_actions = df.groupby(df['Timestamp'].dt.date).size()
    df_cumulative_actions = df_total_actions.cumsum()
    fig, ax = plt.subplots(figsize=chartSize)
    df_cumulative_actions.plot(ax=ax, color='green')
    ax.set_title("Cumulative Total Actions Over Time")
    ax.tick_params(axis='x', rotation=45)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Actions")
    st.pyplot(fig)
    
    
    st.subheader("Total Number of Actions per Day")
    fig, ax = plt.subplots(figsize=chartSize)
    df_total_actions.plot(ax=ax, color='blue')
    ax.set_title("Total Number of Actions per Day")
    ax.tick_params(axis='x', rotation=45)
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Actions")
    st.pyplot(fig)
    
    
    
    # Chart 1: App Usage Over Time
    st.subheader("App Usage Over Time")
    df_grouped_time = df.groupby([df['Timestamp'].dt.date, 'App']).size().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=chartSize)
    for app in df_grouped_time.columns:
        if app == 'GreenAmbassador':
            ax.plot(df_grouped_time.index, df_grouped_time[app], linestyle='--', marker='o', label=app)
        else:
            ax.plot(df_grouped_time.index, df_grouped_time[app], label=app)
    ax.set_title("App Usage Over Time", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Number of Actions", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title="App", title_fontsize=12)
    st.pyplot(fig)
    st.write("This chart shows the daily usage of each app over time, allowing us to see trends and patterns in app popularity. Note that GreenAmbassador data is only available on Mondays.")
    
    # Chart 2: User Engagement Over Time
    st.subheader("User Engagement Over Time")
    
    # Calculate unique receivers per app over time
    df_unique_users = df.groupby(['App', df['Timestamp'].dt.date])['Receiver'].nunique().reset_index()
    df_unique_users.columns = ['App', 'Date', 'Unique_Users']
    
    # Create the plot
    fig, ax = plt.subplots(figsize=chartSize)
    for app in df_unique_users['App'].unique():
        app_data = df_unique_users[df_unique_users['App'] == app]
        ax.plot(app_data['Date'], app_data['Unique_Users'], label=app)
    
    ax.set_title("Unique Users Over Time by App")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Unique Users")
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title="App")
    
    st.pyplot(fig)
    
    st.write("This graph shows the number of unique users for each app over time.")

    # Chart 4: Unique Receivers/Wallets per App
    st.subheader("Unique Receivers/Wallets per App")
    df_unique_receivers = df.groupby('App')['Receiver'].nunique().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=chartSize)
    df_unique_receivers.plot(kind='bar', ax=ax, color=plt.cm.Set2(np.arange(len(df_unique_receivers))))
    ax.set_title("Unique Receivers/Wallets per App", fontsize=16)
    ax.set_xlabel("App", fontsize=12)
    ax.set_ylabel("Unique Receivers/Wallets", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    st.write("This graph shows the number of unique users (wallets) for each app, indicating the size of each app's user base.")
    
    # Chart 3: Total Reward Distributed per App
    st.subheader("Total Reward Distributed per App")
    df_grouped_reward = df.groupby('App')['Reward'].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=chartSize)
    df_grouped_reward.plot(kind='bar', ax=ax, color=plt.cm.Set3(np.arange(len(df_grouped_reward))))
    ax.set_title("Total Reward Distributed per App", fontsize=16)
    ax.set_xlabel("App", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    st.write("This chart displays the total rewards distributed by each app, giving insight into which apps are most generous or popular in terms of rewards.")
    
    
    
    # Chart 5: Average Reward per User per App
    st.subheader("Average Reward per User per App")
    df_avg_reward = df.groupby(['App', 'Receiver'])['Reward'].mean().reset_index()
    fig, ax = plt.subplots(figsize=chartSize)
    sns.boxplot(x='App', y='Reward', data=df_avg_reward, ax=ax, hue='App', legend=False)
    ax.set_title("Average Reward per User per App", fontsize=16)
    ax.set_xlabel("App", fontsize=12)
    ax.set_ylabel("Average Reward", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    st.write("This boxplot visualizes the distribution of average rewards per user for each app, helping to identify which apps tend to give higher or more consistent rewards.")
    
    # Chart 6: Daily Actions per App (Stacked Area)
    st.subheader("Daily Actions per App (Stacked Area)")
    df_daily_actions = df.groupby([df['Timestamp'].dt.date, 'App']).size().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=chartSize)
    df_daily_actions.plot(kind='area', stacked=True, ax=ax, cmap='viridis')
    ax.set_title("Daily Actions per App", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Number of Actions", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title="App", title_fontsize=12)
    st.pyplot(fig)
    st.write("This stacked area chart shows the daily number of actions for each app, allowing us to see how app usage evolves over time and in relation to each other.")
    
    # Chart 7: App Usage by Day of the Week (Heatmap)
    st.subheader("App Usage by Day of the Week (Heatmap)")
    df_heatmap = df.groupby(['DayOfWeek', 'App']).size().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=chartSize)
    sns.heatmap(df_heatmap.T, cmap='YlGnBu', annot=True, fmt='g', ax=ax, cbar_kws={'label': 'Number of Actions'})
    ax.set_title("App Usage by Day of the Week", fontsize=16)
    ax.set_xlabel("Day of Week", fontsize=12)
    ax.set_ylabel("App", fontsize=12)
    st.pyplot(fig)
    st.write("This heatmap visualizes app usage patterns across different days of the week, helping to identify peak usage days for each app. Note that GreenAmbassador data is only available on Mondays.")
    
    # Chart 8: Total Rewards Distributed Over Time
    st.subheader("Total Rewards Distributed Over Time")
    df_rewards_time = df.groupby([df['Timestamp'].dt.date, 'App'])['Reward'].sum().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=chartSize)
    for app in df_rewards_time.columns:
        if app == 'GreenAmbassador':
            ax.plot(df_rewards_time.index, df_rewards_time[app], linestyle='--', marker='o', label=app)
        else:
            ax.plot(df_rewards_time.index, df_rewards_time[app], label=app)
    ax.set_title("Total Rewards Distributed Over Time", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title="App", title_fontsize=12)
    st.pyplot(fig)
    st.write("This chart shows how the total rewards distributed by each app change over time, which can indicate changes in app popularity or reward strategies. Note that GreenAmbassador data is only available on Mondays.")
    
    # Chart 9: First-time Users Over Time
    st.subheader("First-time Users Over Time")
    # Exclude GreenAmbassador app as it's an outlier and messes up the chart
    df_first_time_users = df[df['App'] != 'GreenAmbassador'].groupby([df['FirstSeen'].dt.date, 'App'])['Receiver'].nunique().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=chartSize)
    for app in df_first_time_users.columns:
        ax.plot(df_first_time_users.index, df_first_time_users[app], label=app)
    ax.set_title("First-time Users Over Time", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Number of First-Time Users", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title="App", title_fontsize=12)
    st.pyplot(fig)
    st.write("This graph shows the number of new users joining each app over time, helping to visualize user acquisition trends. Note: GreenAmbassador app is not plotted as it's an outlier/all data on monday. They have been removed to see patterns in remaining apps")
    # Chart 10: Reward Distribution by Day of the Week
    st.subheader("Reward Distribution by Day of the Week")
    fig, ax = plt.subplots(figsize=chartSize)
    sns.boxplot(x='DayOfWeek', y='Reward', data=df, hue='DayOfWeek', legend=False, ax=ax)
    ax.set_title("Reward Distribution by Day of the Week", fontsize=16)
    ax.set_xlabel("Day of Week", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    st.write("This boxplot shows how rewards are distributed across different days of the week, which can reveal patterns in reward allocation. Note that GreenAmbassador data is only available on Mondays.")
    
    # Chart 11: Cumulative Rewards per App Over Time
    st.subheader("Cumulative Rewards per App Over Time")
    df_cumulative_rewards = df.groupby([df['Timestamp'].dt.date, 'App'])['Reward'].sum().groupby('App').cumsum().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=chartSize)
    for app in df_cumulative_rewards.columns:
        ax.plot(df_cumulative_rewards.index, df_cumulative_rewards[app], label=app)
    ax.set_title("Cumulative Rewards per App Over Time", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Reward", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title="App", title_fontsize=12)
    st.pyplot(fig)
    st.write("This chart displays the cumulative rewards distributed by each app over time, showing the total value generated by each app. Note that GreenAmbassador data is only available on Mondays.")
    
    # Chart 12: Actions by Hour of the Day
    st.subheader("Actions by Hour of the Day")
    df_actions_hour = df.groupby(['HourOfDay', 'App']).size().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=chartSize)
    df_actions_hour.plot(ax=ax, linewidth=2)
    ax.set_title("Actions by Hour of the Day", fontsize=16)
    ax.set_xlabel("Hour", fontsize=12)
    ax.set_ylabel("Number of Actions", fontsize=12)
    ax.legend(title="App", title_fontsize=12)
    st.pyplot(fig)
    st.write("This graph shows the number of actions taken in each app throughout the day, revealing peak usage hours for each app.")
    
    # Chart 13: Unique Receivers Growth Over Time
    st.subheader("Unique Receivers Growth Over Time")
    df_unique_receivers_growth = df.groupby([df['Timestamp'].dt.date, 'App'])['Receiver'].nunique().groupby('App').cumsum().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=chartSize)
    for app in df_unique_receivers_growth.columns:
        ax.plot(df_unique_receivers_growth.index, df_unique_receivers_growth[app], label=app)
    ax.set_title("Unique Receivers Growth Over Time", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Unique Receivers", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title="App", title_fontsize=12)
    st.pyplot(fig)
    st.write("This chart shows the cumulative growth of unique users for each app over time, helping to visualize user acquisition and retention trends.")
    
    
    # Chart 14: Actions per App per Receiver
    st.subheader("Actions per App per Receiver")
    df_actions_per_user = df.groupby(['App', 'Receiver']).size().reset_index(name='Actions')
    fig, ax = plt.subplots(figsize=chartSize)
    sns.boxplot(x='App', y='Actions', data=df_actions_per_user, palette="muted", ax=ax)
    ax.set_title("Actions per App per Receiver", fontsize=16)
    ax.set_xlabel("App", fontsize=12)
    ax.set_ylabel("Number of Actions", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    st.write("This boxplot shows the distribution of the number of actions taken by users in each app, indicating user engagement levels across different apps.")
    
    # Chart 15: Time Between First Action and Current Action (Days)
    st.subheader("Time Between First Action and Current Action (Days)")
    df['TimeSinceFirst'] = (df['Timestamp'] - df['FirstSeen']).dt.total_seconds() / (60 * 60 * 24)
    fig, ax = plt.subplots(figsize=chartSize)
    sns.boxplot(x='App', y='TimeSinceFirst', data=df, palette="coolwarm", ax=ax)
    ax.set_title("Time Between First Action and Current Action (Days)", fontsize=16)
    ax.set_xlabel("App", fontsize=12)
    ax.set_ylabel("Days Since First Action", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    st.write("This boxplot visualizes the distribution of time between a user's first action and their subsequent actions for each app, indicating user retention and long-term engagement.")
    
    # Chart 16: Heatmap of Total Rewards by App and Day of the Week
    st.subheader("Heatmap of Total Rewards by App and Day of the Week")
    df_heatmap_rewards = df.groupby([df['DayOfWeek'], 'App'])['Reward'].sum().unstack().fillna(0)
    fig, ax = plt.subplots(figsize=chartSize)
    max_reward = df_heatmap_rewards.max().max()
    norm = mcolors.LogNorm(vmin=1, vmax=max_reward)
    sns.heatmap(df_heatmap_rewards.T, cmap='viridis', norm=norm, annot=True, fmt='.0f', ax=ax, cbar_kws={'label': 'Total Reward'})
    ax.set_title("Total Rewards by App and Day of the Week", fontsize=16)
    ax.set_xlabel("Day of Week", fontsize=12)
    ax.set_ylabel("App", fontsize=12)
    st.pyplot(fig)
    
    st.subheader("Distribution of Days Since First Interaction")
    fig, ax = plt.subplots(figsize=chartSize)
    sns.histplot(df['DaysSinceFirst'], kde=True, bins=30, color='purple', ax=ax)
    ax.set_title("Distribution of Days Since First Interaction", fontsize=16)
    ax.set_xlabel("Days Since First Interaction", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    st.pyplot(fig)
    
    st.subheader("App Usage: Weekdays vs Weekends")
    fig, ax = plt.subplots(figsize=chartSize)
    sns.countplot(x='Weekend', hue='App', data=df, palette='Set3', ax=ax)
    ax.set_title("App Usage: Weekdays vs Weekends", fontsize=16)
    ax.set_xlabel("Is Weekend", fontsize=12)
    ax.set_ylabel("Number of Actions", fontsize=12)
    ax.legend(title="App", title_fontsize=12)
    st.pyplot(fig)
    
    st.subheader("Receiver Retention by App")
    df_retention = df.groupby(['App', 'Receiver']).size().groupby('App').mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=chartSize)
    df_retention.plot(kind='bar', ax=ax, color=plt.cm.Set3(np.arange(len(df_retention))))
    ax.set_title("Average Retention (Actions per Receiver) by App", fontsize=16)
    ax.set_xlabel("App", fontsize=12)
    ax.set_ylabel("Average Actions per Receiver", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    
    st.subheader("Total Reward vs. Unique Receivers per App")
    df_total_reward = df.groupby('App')['Reward'].sum().reset_index()
    df_unique_receivers = df.groupby('App')['Receiver'].nunique().reset_index()
    df_reward_vs_receivers = pd.merge(df_total_reward, df_unique_receivers, on='App')
    
    fig, ax = plt.subplots(figsize=chartSize)
    sns.scatterplot(x='Receiver', y='Reward', data=df_reward_vs_receivers, hue='App', s=100, ax=ax)
    ax.set_title("Total Reward vs. Unique Receivers per App", fontsize=16)
    ax.set_xlabel("Unique Receivers", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.legend(title="App", title_fontsize=12)
    st.pyplot(fig)

    # Churn Analysis
    st.subheader("Churn Analysis")
    st.write("This chart shows the churn rate for each app. Churn rate is calculated as the proportion of users who only interacted with the app once.")
    st.write("We calculate this by first counting the number of unique days each user interacted with each app. Then, we consider a user 'churned' if they only interacted on one day. The churn rate is the number of churned users divided by the total number of users for each app.")
    
    df_churn = df.groupby(['App', 'Receiver'])['Timestamp'].nunique().reset_index()
    df_churn = df_churn.groupby('App').apply(lambda x: x[x['Timestamp'] == 1].shape[0] / x.shape[0])
    fig, ax = plt.subplots(figsize=chartSize)
    df_churn.plot(kind='bar', ax=ax, color=plt.cm.Set3(np.arange(len(df_churn))))
    ax.set_title("Churn Rate by App", fontsize=16)
    ax.set_xlabel("App", fontsize=12)
    ax.set_ylabel("Churn Rate", fontsize=12)
    st.pyplot(fig)

    # User Activity Heatmap
    st.subheader("User Activity Heatmap")
    st.write("This heatmap shows the intensity of user activity across different hours of the day and days of the week.")
    
    pivot = df.pivot_table(values='Receiver', index='DayOfWeek', columns='HourOfDay', aggfunc='count')
    fig, ax = plt.subplots(figsize=(16, 10))  # Increase figure size

    # Use a more visually appealing colormap
    sns.heatmap(pivot, cmap='viridis', norm=norm, ax=ax, annot=True, fmt='g', cbar_kws={'label': 'Number of Actions'})

    ax.set_title("User Activity Heatmap", fontsize=20, pad=20)  # Increase title font size and padding
    ax.set_xlabel("Hour of Day", fontsize=14, labelpad=10)  # Increase label font size and padding
    ax.set_ylabel("Day of Week", fontsize=14, labelpad=10)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0)

    # Adjust colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Number of Actions', fontsize=14, labelpad=10)

    # Improve overall layout
    plt.tight_layout()

    st.pyplot(fig)


    # User Engagement Over Time (Rolling Average)
    st.subheader("User Engagement Over Time (7-day Rolling Average)")
    st.write("This chart shows the 7-day rolling average of daily active users for each app.")
    
    df_daily_users = df.groupby([df['Timestamp'].dt.date, 'App'])['Receiver'].nunique().unstack().fillna(0)
    df_rolling = df_daily_users.rolling(window=7).mean()
    
    fig, ax = plt.subplots(figsize=chartSize)
    for app in df_rolling.columns:
        ax.plot(df_rolling.index, df_rolling[app], label=app)
    ax.set_title("User Engagement Over Time (7-day Rolling Average)", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Number of Active Users", fontsize=12)
    ax.legend(title="App", title_fontsize=12)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Cumulative Unique Users Over Time
    st.subheader("Cumulative Unique Users Over Time")
    st.write("This chart shows the cumulative number of unique users for each app over time.")
    
    df_cumulative_users = df.groupby([df['Timestamp'].dt.date, 'App'])['Receiver'].nunique().unstack().fillna(0).cumsum()
    
    fig, ax = plt.subplots(figsize=chartSize)
    for app in df_cumulative_users.columns:
        ax.plot(df_cumulative_users.index, df_cumulative_users[app], label=app)
    ax.set_title("Cumulative Unique Users Over Time", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Unique Users", fontsize=12)
    ax.legend(title="App", title_fontsize=12)
    plt.xticks(rotation=45)
    st.pyplot(fig)


    # Top 10 Most Active Users
    st.subheader("Top 10 Most Active Users")
    st.write("This chart shows the top 10 most active users across all apps.")

    top_users = df.groupby('Receiver').size().nlargest(10).reset_index(name='ActivityCount')
    fig, ax = plt.subplots(figsize=chartSize)
    sns.barplot(data=top_users, x='Receiver', y='ActivityCount', ax=ax)
    ax.set_title("Top 10 Most Active Users", fontsize=16)
    ax.set_xlabel("User", fontsize=12)
    ax.set_ylabel("Number of Actions", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    # User Activity by App
    st.subheader("User Activity by App")
    st.write("This chart shows the distribution of user activity levels for each app.")

    fig, ax = plt.subplots(figsize=chartSize)
    sns.boxplot(data=df, x='App', y='DaysSinceFirst', ax=ax)
    ax.set_title("User Activity Duration by App", fontsize=16)
    ax.set_xlabel("App", fontsize=12)
    ax.set_ylabel("Days Since First Activity", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    # New vs Returning Users Over Time
    st.subheader("New vs Returning Users Over Time")
    st.write("This chart shows the number of new and returning users over time.")

    df['UserType'] = df.groupby('Receiver')['Timestamp'].transform(lambda x: np.where(x == x.min(), 'New', 'Returning'))
    user_type_over_time = df.groupby([df['Timestamp'].dt.date, 'UserType']).size().unstack().fillna(0)

    fig, ax = plt.subplots(figsize=chartSize)
    user_type_over_time.plot(kind='area', stacked=True, ax=ax)
    ax.set_title("New vs Returning Users Over Time", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Number of Users", fontsize=12)
    ax.legend(title="User Type", title_fontsize=12)
    plt.xticks(rotation=45)
    st.pyplot(fig)
   

# Generate and display the visualizations
create_visualizations()