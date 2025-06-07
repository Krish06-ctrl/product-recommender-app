import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Expanded dataset
data = {
    'User': [
        'Alice', 'Alice', 'Alice', 'Bob', 'Bob', 'Bob', 'Carol', 'Carol', 'Dave', 'Dave',
        'Eve', 'Eve', 'Frank', 'Frank', 'Grace', 'Grace', 'Heidi', 'Heidi', 'Ivan', 'Ivan',
        'Judy', 'Judy', 'Mallory', 'Mallory', 'Niaj', 'Niaj', 'Olivia', 'Olivia', 'Peggy', 'Peggy'
    ],
    'Product': [
        'Laptop', 'Headphones', 'Mouse', 'Laptop', 'Mouse', 'Smartphone', 'Headphones', 'Smartphone', 'Laptop', 'Smartphone',
        'Headphones', 'Mouse', 'Laptop', 'Mouse', 'Headphones', 'Smartphone', 'Laptop', 'Mouse', 'Smartphone', 'Headphones',
        'Laptop', 'Mouse', 'Smartphone', 'Headphones', 'Laptop', 'Smartphone', 'Mouse', 'Headphones', 'Laptop', 'Smartphone'
    ],
    'Rating': [
        5, 3, 4, 4, 5, 2, 2, 5, 3, 4,
        4, 3, 5, 4, 3, 5, 4, 2, 4, 3,
        5, 4, 5, 3, 4, 4, 3, 4, 4, 5
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Create user-item matrix
user_item_matrix = df.pivot_table(index='User', columns='Product', values='Rating').fillna(0)

# User similarity matrix
user_sim_matrix = cosine_similarity(user_item_matrix)
user_sim_df = pd.DataFrame(user_sim_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

# Recommendation logic
def recommend_products(user, user_item_matrix, user_sim_df, top_n=5):
    if user not in user_item_matrix.index:
        return []
    
    # Similarities with all users
    sim_scores = user_sim_df[user].drop(user)
    
    # Products not yet rated by this user
    user_ratings = user_item_matrix.loc[user]
    unrated_products = user_ratings[user_ratings == 0].index
    
    # Score each unrated product
    product_scores = {}
    for product in unrated_products:
        total_sim = 0
        weighted_ratings = 0
        for other_user in sim_scores.index:
            rating = user_item_matrix.loc[other_user, product]
            if rating > 0:
                similarity = sim_scores[other_user]
                weighted_ratings += similarity * rating
                total_sim += similarity
        if total_sim > 0:
            product_scores[product] = weighted_ratings / total_sim
    
    # Sort by highest predicted rating
    sorted_products = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)
    recommended = [product for product, score in sorted_products[:top_n]]
    
    return recommended
# Streamlit app
st.title("Product Recommendation System")

# User selection
selected_user = st.selectbox("Select a user", user_item_matrix.index)

# Show recommendations
if st.button("Get Recommendations"):
    recommendations = recommend_products(selected_user, user_item_matrix, user_sim_df)
    if recommendations:
        st.write(f"Recommended products for {selected_user}:")
        for product in recommendations:
            st.write(f"- {product}")
    else:
        st.write(f"No recommendations available for {selected_user}.")
