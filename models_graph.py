import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

#Used to load Data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Used to load your data file
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

# Generate predictions for test set (for evaluation)
y_pred_cal = calories_pipeline.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test_cal, y_pred_cal)
mse = mean_squared_error(y_test_cal, y_pred_cal)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_cal, y_pred_cal)

# Save the model and evaluation metrics
model_data = {
    'pipeline': calories_pipeline,
    'metrics': {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    },
    'feature_importance': calories_pipeline.named_steps['regressor'].feature_importances_,
    'X_test': X_test,
    'y_test': y_test_cal,
    'y_pred': y_pred_cal
}
joblib.dump(model_data, 'calories_model_with_evaluation.pkl')
joblib.dump(calories_pipeline, 'calories_model.pkl')

# 3. Streamlit Application
st.set_page_config(page_title='Personal Fitness Tracker', page_icon=':runner:', layout='wide')

# Load models with evaluation data
try:
    model_data = joblib.load('calories_model_with_evaluation.pkl')
    calories_pipeline = model_data['pipeline']
    metrics = model_data['metrics']
    feature_importance = model_data['feature_importance']
    X_test_saved = model_data['X_test']
    y_test_saved = model_data['y_test']
    y_pred_saved = model_data['y_pred']
except:
    calories_pipeline = joblib.load('calories_model.pkl')

# Apply custom CSS
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .user-input-table { padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .stSlider>div>div>div>div { background: red; }
    .metrics-card {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title('Personal Fitness Tracker')

# Create tabs for different sections
tabs = st.tabs(["Live Predictions", "Model Performance", "Feature Analysis"])

with tabs[0]:  # Live Predictions Tab
    # Sidebar for user input
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

    # Display user input in formatted table at the top
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
        
        # Base wave plot
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

    # Create all visualizations
    col1, col2 = st.columns(2)
    with col1:
        plot1 = create_live_wave_plot('Weight (kg)', 'Calories_Burned', 
                                    'Weight vs Calories Burned',
                                    input_df['Weight (kg)'].values[0], 
                                    predicted_calories)
        st.pyplot(plot1)

    with col2:
        plot2 = create_live_wave_plot('Age', 'Calories_Burned',
                                    'Age vs Calories Burned',
                                    input_df['Age'].values[0],
                                    predicted_calories)
        st.pyplot(plot2)
    
    col1, col2 = st.columns(2)
    with col1:
        plot3 = create_live_wave_plot('Session_Duration (hours)', 'Calories_Burned',
                                    'Session Duration vs Calories Burned',
                                    input_df['Session_Duration (hours)'].values[0],
                                    predicted_calories)
        st.pyplot(plot3)
        
    with col2:
        plot4 = create_live_wave_plot('Avg_BPM', 'Calories_Burned',
                                    'Average BPM vs Calories Burned',
                                    input_df['Avg_BPM'].values[0],
                                    predicted_calories)
        st.pyplot(plot4)

with tabs[1]:  # Model Performance Tab
    st.header("Model Evaluation")
    
    try:
        # Display metrics in a nice grid
        st.subheader("Model Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metrics-card">', unsafe_allow_html=True)
            st.metric("MAE", f"{metrics['mae']:.2f}")
            st.markdown("Mean Absolute Error")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metrics-card">', unsafe_allow_html=True)
            st.metric("MSE", f"{metrics['mse']:.2f}")
            st.markdown("Mean Squared Error")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="metrics-card">', unsafe_allow_html=True)
            st.metric("RMSE", f"{metrics['rmse']:.2f}")
            st.markdown("Root Mean Squared Error")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col4:
            st.markdown('<div class="metrics-card">', unsafe_allow_html=True)
            st.metric("R² Score", f"{metrics['r2']:.3f}")
            st.markdown("Coefficient of Determination")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Actual vs Predicted Plot
        st.subheader("Actual vs Predicted Calories")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the scatter points
        scatter = ax.scatter(y_test_saved, y_pred_saved, 
                  alpha=0.6, 
                  edgecolor='w',
                  c=abs(y_test_saved-y_pred_saved), 
                  cmap='viridis')
        
        # Add the perfect prediction line
        max_val = max(y_test_saved.max(), y_pred_saved.max())
        min_val = min(y_test_saved.min(), y_pred_saved.min())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.7, label='Perfect Prediction')
        
        # Add a colorbar to show the error magnitude
        cbar = plt.colorbar(scatter)
        cbar.set_label('Absolute Error')
        
        # Styling
        ax.set_xlabel('Actual Calories Burned')
        ax.set_ylabel('Predicted Calories Burned')
        ax.set_title('Actual vs Predicted Calories Burned', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend()
        
        st.pyplot(fig)
        
        # Residual Plot
        st.subheader("Residual Analysis")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate residuals
        residuals = y_test_saved - y_pred_saved
        
        # Plot residuals
        ax.scatter(y_pred_saved, residuals, alpha=0.6, edgecolor='w')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        
        # Styling
        ax.set_xlabel('Predicted Calories Burned')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.spines[['top', 'right']].set_visible(False)
        
        st.pyplot(fig)
        
        # Error Distribution
        st.subheader("Error Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram of residuals
        sns.histplot(residuals, kde=True, ax=ax)
        
        # Styling
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Prediction Errors', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.spines[['top', 'right']].set_visible(False)
        
        st.pyplot(fig)
        
    except:
        st.warning("Model evaluation metrics are not available. Please train the model first to view performance metrics.")

with tabs[2]:  # Feature Analysis Tab
    st.header("Feature Importance and Analysis")
    
    try:
        # Feature importance plot
        st.subheader("Feature Importance")
        
        # Get feature names after one-hot encoding
        numeric_features = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM', 'Session_Duration (hours)', 'Workout_Frequency (days/week)']
        categorical_features = ['Gender']
        
        # Create transformer to get feature names
        ct = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numeric_features),
                ('cat', OneHotEncoder(), categorical_features)
            ])
        
        # Fit the transformer to get feature names
        ct.fit(X)
        
        # Get feature names after transformation
        ohe = ct.named_transformers_['cat']
        categorical_feature_names = ohe.get_feature_names_out(['Gender'])
        all_features = np.concatenate([numeric_features, categorical_feature_names])
        
        # Create feature importance dataframe
        feature_importance_df = pd.DataFrame({
            'Feature': all_features,
            'Importance': feature_importance
        }).sort_values(by='Importance', ascending=False)
        
        # Plot feature importance
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax)
        
        # Styling
        ax.set_title('Feature Importance for Calories Burned Prediction', fontsize=14)
        ax.grid(True, axis='x', linestyle='--', alpha=0.6)
        ax.spines[['top', 'right']].set_visible(False)
        
        st.pyplot(fig)
        
        # Feature correlation heatmap
        st.subheader("Feature Correlation")
        
        # Calculate correlation matrix
        corr_matrix = fitness_data[numeric_features + ['Calories_Burned']].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.zeros_like(corr_matrix)
        mask[np.triu_indices_from(mask)] = True
        
        # Plot correlation heatmap
        sns.heatmap(corr_matrix, 
                    mask=mask,
                    annot=True, 
                    fmt=".2f", 
                    cmap='coolwarm', 
                    center=0,
                    square=True,
                    linewidths=.5,
                    cbar_kws={'shrink': .8},
                    ax=ax)
        
        # Styling
        ax.set_title('Feature Correlation Heatmap', fontsize=14)
        
        st.pyplot(fig)
        
        # Pairplot for key features
        st.subheader("Relationship Between Key Features")
        
        # Select top features based on importance and add target
        top_features = feature_importance_df.head(3)['Feature'].tolist()
        top_features = [f for f in top_features if f in numeric_features]  # Keep only numeric features
        plot_features = top_features + ['Calories_Burned']
        
        # Create pairplot with selected features
        fig = sns.pairplot(fitness_data[plot_features], 
                         height=2.5, 
                         corner=True, 
                         diag_kind='kde',
                         plot_kws={'alpha': 0.6, 'edgecolor': 'k', 'linewidth': 0.5})
        
        fig.fig.suptitle('Pairwise Relationships Between Top Features', y=1.02, fontsize=16)
        st.pyplot(fig.fig)
        
    except Exception as e:
        st.warning(f"Feature analysis data is not available. Please train the model first to view feature analysis.")
        st.error(f"Error: {e}")