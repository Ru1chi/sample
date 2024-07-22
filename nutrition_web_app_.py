# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:44:53 2024
1.2.2
@author: sr322
"""





import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle

# Function to load recipe data from an Excel file
def load_recipe_data():
    return pd.read_excel(r'C:\Users\sr322\OneDrive\Desktop\ML_Model_Deployment\Nutrition in budget\recipeeees.xlsx')  # Adjust file path as necessary

# Function to load model and scaler
def load_model():
    with open(r'C:\Users\sr322\OneDrive\Desktop\ML_Model_Deployment\Nutrition in budget\kmeans_model11.sav', 'rb') as model_file:
        model = pickle.load(model_file)
    with open(r'C:\Users\sr322\OneDrive\Desktop\ML_Model_Deployment\Nutrition in budget\scaler_model11.sav', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

# Load the data and model
model, scaler = load_model()
data = load_recipe_data()

# Define ingredient categories for dietary preferences
dietary_ingredients = {
    'Chicken': ['chicken', 'chicken breast', 'chicken thigh', 'chicken wing'],
    'Beef': ['beef', 'steak', 'ground beef', 'beef ribs'],
    'Vegetarian': ['tofu', 'tempeh', 'vegetables', 'beans', 'lentils', 'cheese', 'eggs'],
    'Vegan': ['tofu', 'tempeh', 'vegetables', 'beans', 'lentils', 'nuts', 'seeds'],
    'Fish': ['fish', 'salmon', 'tuna', 'cod', 'tilapia']
}

# Function to categorize recipes based on dietary preferences
def categorize_recipe(ingredients, dietary_preferences):
    for preference in dietary_preferences:
        for ingredient in dietary_ingredients[preference]:
            if ingredient.lower() in ingredients.lower():
                return True
    return False

# User Inputs
st.title('Personalized Meal Plan Generator')

# Collect dietary preferences from the user
dietary_preference = st.multiselect(
    'Select Dietary Preferences',
    options=['Chicken', 'Beef', 'Vegetarian', 'Vegan', 'Fish'],
    default=['Vegetarian']
)

# Collect health goal from the user
health_goal = st.selectbox(
    'Select Health Goal',
    options=['Weight Loss', 'Muscle Gain', 'Maintain Weight']
)

# Collect budget and meal frequency from the user
budget = st.slider(
    'Select Your Budget ($ per meal)',
    min_value=1, max_value=20, value=10
)

meals_per_day = st.slider(
    'Meals per Day',
    min_value=1, max_value=5, value=3
)

days_per_week = st.slider(
    'Days per Week',
    min_value=1, max_value=7, value=7
)

# Filter data based on dietary preferences
filtered_data = data[data['ingredients'].apply(categorize_recipe, args=(dietary_preference,))]

# Define nutritional goals
if health_goal == 'Weight Loss':
    calorie_limit = 500
elif health_goal == 'Muscle Gain':
    calorie_limit = 700
else:
    calorie_limit = 600

# Prepare data for model prediction
if not filtered_data.empty:
    X = filtered_data[['calories', 'protein', 'carbs', 'fat', 'price($)']]
    X_scaled = scaler.transform(X)

    target = [[calorie_limit, 0, 0, 0, budget]]
    target_scaled = scaler.transform(target)

    # Fit KMeans model on the scaled filtered data
     #model = KMeans(n_clusters=model.n_clusters, n_init='auto')  # Ensure n_init is correctly set
    model = KMeans(n_clusters=min(model.n_clusters, X_scaled.shape[0]), n_init=10)

    model.fit(X_scaled)
    filtered_data['cluster'] = model.labels_

    # Get cluster label for the target
    target_cluster = model.predict(target_scaled)[0]

    # Filter data based on the cluster label of the target
    target_cluster_data = filtered_data[filtered_data['cluster'] == target_cluster]

    # Get recommendations from the filtered data
    recommendations = target_cluster_data.sample(n=meals_per_day * days_per_week, replace=True)
else:
    recommendations = pd.DataFrame()  # Empty DataFrame

# Generate Weekly Meal Plan
if not recommendations.empty:
    st.subheader('Weekly Meal Plan')
    for idx, row in recommendations.iterrows():
        st.write(f"**{row['recipe_name']}**")
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
else:
    st.write("No recommendations available, hence no nutritional analysis.")

# Generate and Display Shopping List
if not recommendations.empty:
    st.subheader('Shopping List')
    ingredients_list = recommendations['ingredients'].str.split(', ').explode().value_counts()
    for ingredient, count in ingredients_list.items():
        st.write(f"{ingredient} (x{count})")
else:
    st.write("No recommendations available, hence no shopping list.")
