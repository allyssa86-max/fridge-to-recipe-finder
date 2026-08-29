# Fridge-to-Recipe Finder

## Project Goal
A recipe recommendation system that suggests recipes based on ingredients 
available in the user's fridge. Users photograph individual ingredients, 
which are classified using a fine-tuned CNN (transfer learning), then 
matched to relevant recipes using a RAG pipeline.

## Business Problem
Help users daily mental load by figuring out every day what to cook, with ingredients already available in their fridge or pantry. 


## Dataset
- Recipes:"https://www.kaggle.com/datasets/wilmerarltstrmberg/recipe-dataset-over-2m" subset ~500-1000 recipes
- Ingredient images: Kaggle "Fruits and Vegetables Image Recognition" 
  or similar for CNN fine-tuning

## Methodology
1. **EDA:** ingredient frequency, cuisine distribution, recipe length
2. **Deep Learning (CNN + Transfer Learning):** fine-tuned model 
   (MobileNet/ResNet) for single-ingredient image classification, 
   evaluated with accuracy/confusion matrix
3. **Gen AI (RAG):** recipes embedded and stored in ChromaDB; LangChain 
   retrieval matches detected ingredients to relevant recipes; LLM 
   (Ollama) generates the final suggestion
4. **Application:** Streamlit interface: photo upload → CNN → RAG → 
   recipe suggestions

## Key Findings
[Fill in after EDA/modeling]

## Model Performance
[Fill in: CNN accuracy, confusion matrix interpretation]

## Challenges & Decisions
[to be filled in]

## How to Run
\`\`\`
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## Tech Stack
Python, pandas, TensorFlow/Keras, scikit-learn, LangChain, ChromaDB, 
Ollama, Streamlit, matplotlib/seaborn

## Author
Dewi Wirokarto

