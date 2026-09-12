# Fridge-to-Recipe Finder

## Project Goal
A recipe recommendation system that suggests recipes based on ingredients 
available in the user's fridge. Users enter their available 
ingredients as text, which are matched to relevant recipes using TF-IDF 
and cosine similarity.

## Business Problem
Help users daily mental load by figuring out every day what to cook, with ingredients already available in their fridge or pantry. 


## Dataset
- Recipes:"https://www.kaggle.com/datasets/wilmerarltstrmberg/recipe-dataset-over-2m" 

## Methodology
1. **EDA:** ingredient frequency analysis, K-Means clustering on ingredients 
   (TF-IDF) to identify recipe categories, comparative clustering attempt on 
   cooking directions
2. **Machine Learning:** Logistic Regression classifier to predict recipe 
   category based on ingredients, evaluated with accuracy and confusion matrix
3. **Matching function:** TF-IDF vectorization + cosine similarity to match 
   user-entered ingredients to the most relevant recipes
4. **Application:** Streamlit interface — text input → matching function → 
   recipe suggestions with full ingredient/direction details

## Ingredient-based clustering
Using K-Means (k=6) on TF-IDF-vectorized ingredients, recipes were grouped 
into six interpretable categories: Soups, Baking, Juices, Savory/meat dishes, 
Cheesy dishes, and Desserts.

## Key Findings
The recipe dataset contains over 2M recipes from 28 different sites. 
The dataset does luckily provide me of a NER column, which stands for Named Entity Recognition. 
And is a technology within NLP to scan the text and take the words to assign them to their categories.
This is used as the basis for matching and clustering. 

## Model Performance
A Logistic Regression classifier was trained to predict a recipe's cluster 
category (from the K-Means clustering above) based on its TF-IDF-vectorized 
ingredients. 

The model achieved an accuracy score of 93.4%, which is very strong for a 6-class classification task. 
- 16 recipes that were actually cluster 2(Juices), were predicted as cluster 3(savory dishes)
- 5 recipes that were classified as cluster 0(soups), were predicted as cluster 3(savory dishes)
- Cluster 1 (Baking), cluster 4(Cheesy) and cluster 5(Desserts) show strong predictions. 

## Challenges & Decisions
- The biggest challenge during this project was time. Given the constraints, 
  a text-input based recipe finder was chosen over a photo-input version 
  (CNN + Gen AI/RAG pipeline), in order to fully complete and polish the 
  must-have requirements (data, EDA, and end-user interactivity).
- Directions-based clustering was explored as an alternative to ingredient-based 
  clustering but showed less distinctive results, likely due to shared generic 
  cooking vocabulary across recipe types.
- The NER-field (with the pre-extracted ingredient names) in the dataset is not always complete. 
  Some recipes show only a part of the ingredients, even though the full ingredient list is present in the `ingredients` column. 
  This is a limitation of the dataset, and a new annotation of the dataset is out of scope. 

## How to Run
\`\`\`
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## Tech Stack
Python, pandas, scikit-learn, Streamlit, matplotlib

## Next Steps
Given more time, the following extensions would strengthen the project:
- **Photo-based ingredient input:** using a fine-tuned CNN (transfer learning) 
  to classify ingredients from a photo, removing the need for manual text entry
- **Gen AI component:** a RAG pipeline (ChromaDB + LangChain + local LLM via 
  Ollama) to generate more natural, context-aware recipe suggestions and 
  substitution advice
- **Recipe length / complexity analysis:** as an additional EDA dimension

## Author
Dewi Wirokarto

