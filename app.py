import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import pickle

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("C:/Users\sr322/Downloads/recipees_with_all_clusters.csv")

# Function to categorize recipes based on dietary preferences
def categorize_recipe(ingredients, dietary_preference):
    ingredients_lower = ingredients.lower()
    if dietary_preference == 'Vegetarian':
        return not any(meat in ingredients_lower for meat in ['chicken', 'beef', 'fish', 'shrimp', 'steak', 'lamb'])
    elif dietary_preference == 'Vegan':
        return not any(dairy_or_meat in ingredients_lower for dairy_or_meat in ['chicken', 'beef', 'fish', 'shrimp', 'steak', 'lamb', 'cheese', 'milk', 'yogurt', 'egg', 'butter', 'cream'])
    elif dietary_preference == 'Fish':
        return 'fish' in ingredients_lower or 'shrimp' in ingredients_lower
    elif dietary_preference == 'Non-Veg':
        return any(meat in ingredients_lower for meat in ['chicken', 'beef', 'fish', 'shrimp', 'steak', 'lamb'])
    return False

# Load data
df = load_data()

# Sidebar inputs
st.sidebar.title('Personalized Meal Plan Generator')
dietary_preference = st.sidebar.selectbox(
    'Select Dietary Preference',
    options=['Vegetarian', 'Vegan', 'Non-Veg', 'Fish']
)

health_goal = st.sidebar.selectbox(
    'Select Health Goal',
    options=['Weight Loss', 'Muscle Gain', 'Maintain Weight']
)

meals_per_day = st.sidebar.slider(
    'Meals per Day',
    min_value=1, max_value=5, value=3
)

days_per_week = st.sidebar.slider(
    'Days per Week',
    min_value=1, max_value=7, value=7
)

budget = st.sidebar.slider(
    'Budget ($ per meal)',
    min_value=1, max_value=20, value=10
)

# Filter data based on dietary preferences
filtered_data = df[df['ingredients'].apply(lambda x: categorize_recipe(x, dietary_preference))]

# Define nutritional goals
if health_goal == 'Weight Loss':
    calorie_limit = 500
elif health_goal == 'Muscle Gain':
    calorie_limit = 700
else:
    calorie_limit = 600

# Prepare data for model prediction
X = filtered_data[['calories', 'protein', 'carbs', 'fat', 'price($)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Load SVM model
with open('C:/Users/sr322/Downloads/svm_model.sav', 'rb') as f:
    svm_model = pickle.load(f)

# Predict clusters
filtered_data['Cluster_Label'] = svm_model.predict(filtered_data[['PCA1', 'PCA2']])

# Filter data based on the cluster label and budget
target_cluster_data = filtered_data[(filtered_data['price($)'] <= budget) & (filtered_data['calories'] <= calorie_limit)]

# Get recommendations from the filtered data
recommendations = target_cluster_data.sample(n=meals_per_day * days_per_week, replace=True)

# Generate Weekly Meal Plan
st.title('Weekly Meal Plan')
if not recommendations.empty:
    for idx, row in recommendations.iterrows():
        st.write(f"*{row['recipe_name']}*")
        st.write(f"Ingredients: {row['ingredients']}")
        st.write(f"Calories: {row['calories']}, Protein: {row['protein']}g, Carbs: {row['carbs']}g, Fat: {row['fat']}g")
        st.write(f"Price: ${row['price($)']:.2f}")
        st.write("---")
else:
    st.write("No recommendations available based on the provided preferences.")

# Display Nutritional Analysis
if not recommendations.empty:
    total_calories = recommendations['calories'].sum()
    total_protein = recommendations['protein'].sum()
    total_carbs = recommendations['carbs'].sum()
    total_fat = recommendations['fat'].sum()
    total_cost = recommendations['price($)'].sum()
    
    st.subheader('Nutritional Analysis')
    st.write(f"Total Calories: {total_calories}")
    st.write(f"Total Protein: {total_protein}g")
    st.write(f"Total Carbs: {total_carbs}g")
    st.write(f"Total Fat: {total_fat}g")
    st.write(f"Total Cost: ${total_cost:.2f}")

# Generate and Display Shopping List
if not recommendations.empty:
    st.subheader('Shopping List')
    ingredients_list = recommendations['ingredients'].str.split(', ').explode().value_counts()
    for ingredient, count in ingredients_list.items():
        st.write(f"{ingredient} (x{count})")
else:
    st.write("No recommendations available, hence no shopping list.")