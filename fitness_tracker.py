import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# used to Load Data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Used to Load your data file
fitness_data = load_data('gym_members_exercise_tracking_synthetic_data.csv')

# Data Preprocessing
fitness_data.dropna(inplace=True)

# Define features and target variables
features = ['Age', 'Gender', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM', 'Session_Duration (hours)', 'Workout_Frequency (days/week)']
X = fitness_data[features]
y_calories = fitness_data['Calories_Burned']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM', 'Session_Duration (hours)', 'Workout_Frequency (days/week)']),
        ('cat', OneHotEncoder(), ['Gender'])
    ])

# Split data into training and testing sets
X_train, X_test, y_train_cal, y_test_cal = train_test_split(X, y_calories, test_size=0.2, random_state=42)

# Create pipeline for the calories model
calories_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

# Train the model
calories_pipeline.fit(X_train, y_train_cal)

# Save the model
joblib.dump(calories_pipeline, 'calories_model.pkl')

# 3. Streamlit Application
st.set_page_config(page_title='Personal Fitness Tracker', page_icon=':runner:', layout='wide')

# Load models
calories_pipeline = joblib.load('calories_model.pkl')

# Apply custom CSS
st.markdown("""
    <style>
    .main { background-color: green; }
    .user-input-table { padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .stSlider>div>div>div>div { background: red; }
    .metric-card {
        background-color: #262730;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #aaaaaa;
    }
    .metric-comparison {
        font-size: 14px;
        color: #22;
    }
    .comparison-higher { color: #e63946; }
    .comparison-lower { color: #2a9d8f; }
    </style>
    """, unsafe_allow_html=True)

st.title('Personal Fitness Tracker')

# Sidebar used for users input
st.sidebar.header('Live Input Controls')
def user_input_features():
    age = st.sidebar.slider('Age', 18, 100, 30)
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    weight_kg = st.sidebar.slider('Weight (kg)', 30, 150, 70)
    height_m = st.sidebar.slider('Height (m)', 1.5, 2.2, 1.75)
    max_bpm = st.sidebar.slider('Max BPM', 60, 200, 150)
    avg_bpm = st.sidebar.slider('Avg BPM', 60, 200, 120)
    resting_bpm = st.sidebar.slider('Resting BPM', 40, 100, 60)
    session_duration = st.sidebar.slider('Session Duration (hours)', 0.5, 5.0, 1.0)
    workout_frequency = st.sidebar.slider('Workout Frequency (days/week)', 1, 7, 3)
    
    return pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Weight (kg)': [weight_kg],
        'Height (m)': [height_m],
        'Max_BPM': [max_bpm],
        'Avg_BPM': [avg_bpm],
        'Resting_BPM': [resting_bpm],
        'Session_Duration (hours)': [session_duration],
        'Workout_Frequency (days/week)': [workout_frequency]
    })

input_df = user_input_features()

# Calculate percentile 
def calculate_percentile(metric_name, user_value, gender_filter=False):
    if gender_filter:
        filtered_data = fitness_data[fitness_data['Gender'] == input_df['Gender'].values[0]]
    else:
        filtered_data = fitness_data
        
    percentile = (filtered_data[metric_name] < user_value).mean() * 100
    return percentile

# Get percentiles 
age_percentile = calculate_percentile('Age', input_df['Age'].values[0])
weight_percentile = calculate_percentile('Weight (kg)', input_df['Weight (kg)'].values[0], gender_filter=True)
height_percentile = calculate_percentile('Height (m)', input_df['Height (m)'].values[0], gender_filter=True)
max_bpm_percentile = calculate_percentile('Max_BPM', input_df['Max_BPM'].values[0])
avg_bpm_percentile = calculate_percentile('Avg_BPM', input_df['Avg_BPM'].values[0])
resting_bpm_percentile = calculate_percentile('Resting_BPM', input_df['Resting_BPM'].values[0])
duration_percentile = calculate_percentile('Session_Duration (hours)', input_df['Session_Duration (hours)'].values[0])
frequency_percentile = calculate_percentile('Workout_Frequency (days/week)', input_df['Workout_Frequency (days/week)'].values[0])

# Display user input in table at the top
st.subheader('Current User Input Summary')
input_table = input_df.T.reset_index()
input_table.columns = ['Metric', 'Value']

# Format specific columns
input_table['Value'] = input_table.apply(lambda x: 
    f"{x['Value']:.1f}" if x['Metric'] in ['Session_Duration (hours)', 'Weight (kg)', 'Height (m)'] else
    f"{x['Value']:.0f}" if x['Metric'] != 'Gender' else x['Value'], axis=1)

# Display in two columns for better layout
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="user-input-table">
        <h4 style='margin-bottom: 1rem;'>User Metrics</h4>
    """, unsafe_allow_html=True)
    st.table(input_table)
    st.markdown("</div>", unsafe_allow_html=True)

# Generate predictions
predicted_calories = calories_pipeline.predict(input_df)[0]

# Display predictions
st.subheader('Live Predictions:')
st.metric("🔥 Predicted Calories Burned", f"{predicted_calories:.2f} kcal")

# Create dynamic visualizations
st.subheader('Live Data Visualizations')

# Create combined dataset for plotting
combined_data = pd.concat([fitness_data, input_df], ignore_index=True)

# Create animated wave plots
def create_live_wave_plot(x_feature, y_feature, title, user_value, prediction):
    fig, ax = plt.subplots(figsize=(10, 4))
    
    sns.lineplot(
        data=combined_data.sort_values(by=x_feature),
        x=x_feature,
        y=y_feature,
        hue='Gender',
        ci=None,
        estimator=None,
        lw=2,
        alpha=0.3,
        ax=ax
    )
    
    # Add live user point
    ax.scatter(
        user_value,
        prediction,
        c='red',
        s=100,
        edgecolor='black',
        label='Your Current Input',
        zorder=5
    )
    
    # Add trend line
    sns.regplot(
        x=x_feature,
        y=y_feature,
        data=combined_data,
        scatter=False,
        color='blue',
        line_kws={'alpha': 0.5},
        ax=ax
    )
    
    # Styling
    ax.set_title(f"{title} - Live Update", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend()
    return fig

# Display comparison metrics
st.subheader('How You Compare to Others')

# Create comparison metrics in a grid
col1, col2 = st.columns(2)

# Function to display comparison metric with formatting based on percentile
def display_comparison_metric(col, title, value, percentile, unit="", higher_is_better=False):
    comparison_class = "comparison-higher" if (higher_is_better and percentile > 50) or (not higher_is_better and percentile < 50) else "comparison-lower"
    
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value} {unit}</div>
        <div class="metric-comparison">
            You are <span class="{comparison_class}">higher than {percentile:.1f}%</span> of users
        </div>
    </div>
    """, unsafe_allow_html=True)

# Column 1 metrics
display_comparison_metric(col1, "Age", f"{input_df['Age'].values[0]:.0f}", age_percentile, "years", higher_is_better=False)
display_comparison_metric(col1, "Weight", f"{input_df['Weight (kg)'].values[0]:.1f}", weight_percentile, "kg", higher_is_better=False)
display_comparison_metric(col1, "Height", f"{input_df['Height (m)'].values[0]:.2f}", height_percentile, "m", higher_is_better=True)
display_comparison_metric(col1, "Exercise Duration", f"{input_df['Session_Duration (hours)'].values[0]:.1f}", duration_percentile, "hours", higher_is_better=True)

# Column 2 metrics
display_comparison_metric(col2, "Max Heart Rate", f"{input_df['Max_BPM'].values[0]:.0f}", max_bpm_percentile, "BPM", higher_is_better=False)
display_comparison_metric(col2, "Average Heart Rate", f"{input_df['Avg_BPM'].values[0]:.0f}", avg_bpm_percentile, "BPM", higher_is_better=False) 
display_comparison_metric(col2, "Resting Heart Rate", f"{input_df['Resting_BPM'].values[0]:.0f}", resting_bpm_percentile, "BPM", higher_is_better=False)
display_comparison_metric(col2, "Workout Frequency", f"{input_df['Workout_Frequency (days/week)'].values[0]:.0f}", frequency_percentile, "days/week", higher_is_better=True)

# Create visualization plots
plot1 = create_live_wave_plot('Weight (kg)', 'Calories_Burned', 
                             'Weight vs Calories Burned',
                             input_df['Weight (kg)'].values[0], 
                             predicted_calories)

plot2 = create_live_wave_plot('Age', 'Calories_Burned',
                             'Age vs Calories Burned',
                             input_df['Age'].values[0],
                             predicted_calories)

# Display plots in columns
col1, col2 = st.columns(2)
with col1:
    st.pyplot(plot1)

with col2:
    st.pyplot(plot2)

# Add comparative distribution plots
st.subheader('Your Metrics Compared to Population Distribution')

def create_distribution_plot(feature, user_value, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot distribution
    sns.histplot(fitness_data[feature], kde=True, ax=ax)
    
    # Add user's value as vertical line
    ax.axvline(user_value, color='red', linestyle='--', linewidth=2, label='Your Value')
    
    # Add text annotation showing percentile
    percentile = (fitness_data[feature] < user_value).mean() * 100
    ax.text(
        user_value, 
        ax.get_ylim()[1] * 0.9, 
        f"  You: {user_value:.1f}\n  Top {100-percentile:.1f}%" if percentile > 50 else f"  You: {user_value:.1f}\n  Bottom {percentile:.1f}%",
        verticalalignment='top',
        color='red',
        fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.5')
    )
    
    # Styling
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend()
    
    return fig

# Create distribution plots for key metrics
col1, col2 = st.columns(2)
with col1:
    st.pyplot(create_distribution_plot(
        'Session_Duration (hours)', 
        input_df['Session_Duration (hours)'].values[0],
        'Your Exercise Duration vs Population'
    ))

with col2:
    st.pyplot(create_distribution_plot(
        'Resting_BPM', 
        input_df['Resting_BPM'].values[0],
        'Your Resting Heart Rate vs Population'
    ))