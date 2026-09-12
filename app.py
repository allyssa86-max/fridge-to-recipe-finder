import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def load_and_prepare_data():
    df = pd.read_csv('./data/recipes_data.csv')
    df['NER_parsed'] = df['NER'].apply(ast.literal_eval)
    df['NER_text'] = df['NER_parsed'].apply(lambda x: ' '.join(x))
    
    tfidf_matching = TfidfVectorizer()
    tfidf_matrix_matching = tfidf_matching.fit_transform(df['NER_text'])
    
    return df, tfidf_matching, tfidf_matrix_matching

def find_matching_recipes(user_ingredients, df, tfidf_matching, tfidf_matrix_matching, top_n=5):
    user_vector = tfidf_matching.transform([user_ingredients])
    similarities = cosine_similarity(user_vector, tfidf_matrix_matching)
    top_indices = similarities[0].argsort()[::-1][:top_n]
    top_scores = similarities[0][top_indices]
    
    results = df.iloc[top_indices][['title', 'NER_text']].copy()
    results['similarity_score'] = top_scores
    return results

# page setup

st.set_page_config(page_title="Fridge Recipe Finder", layout="wide")

st.title("What's in my fridge?")
st.markdown("Find recipes based on the ingredients you have in your fridge or pantry")

df, tfidf_matching, tfidf_matrix_matching = load_and_prepare_data()

# ---Sidebar Layout ---

st.sidebar.header("Ingredients")
user_input = st.sidebar.text_input("Tell me, what are we working with? (e.g. chicken, rice):")
find_button = st.sidebar.button("Find recipes")


# ---Recipes page ---

if find_button:
    st.session_state['results'] = find_matching_recipes(user_input, df, tfidf_matching, tfidf_matrix_matching) if user_input else None
    if not user_input:
        st.warning("Please enter at least one ingredient.")

if 'results' in st.session_state and st.session_state['results'] is not None:
    results = st.session_state['results']
    
    header_a, header_b = st.columns([2, 3])
    with header_a:
        st.markdown("**Recipe**")
    with header_b:
        st.markdown("**Ingredients**")
    
    for idx, row in results.iterrows():
        col_a, col_b = st.columns([2, 3])
        with col_a:
            st.write(row['title'].strip())
        with col_b:
            preview_ingredients = row['NER_text'].split()[:5]
            st.write(", ".join(preview_ingredients))
    
    options = ["-- Select a recipe --"] + list(results['title'])
    selected_title = st.selectbox("Choose a recipe to see full details:", options)
    
    if selected_title != "-- Select a recipe --":
        selected_recipe = df[df['title'] == selected_title].iloc[0]
        
        st.subheader(selected_title)
        
        st.markdown("**Ingredients:**")
        ingredients_list = ast.literal_eval(selected_recipe['ingredients'])
        for ingredient in ingredients_list:
            st.write(f"- {ingredient}")
        
        st.markdown("**Directions:**")
        directions_list = ast.literal_eval(selected_recipe['directions'])
        for step in directions_list:
            st.write(f"- {step}")

        link = selected_recipe['link']
        if not link.startswith('http'):
            link = 'https://' + link
        st.markdown(f"[View original recipe]({link})")

