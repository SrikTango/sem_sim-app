#Import libraries and modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from scipy.spatial.distance import cosine
#from wordcloud import STOPWORDS
import gensim
from gensim.models import word2vec
#import pickle
from nltk.tokenize import sent_tokenize
import streamlit as st

# Load the pre-trained model from Google News corpus
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)  


#Function for performing text pre=processing
def preprocess_text(para):
    preprocessed_text = []
    for sent in range(len(para)):
        # Replace newline characters, quotes, etc.
        sent_text = re.sub('\\r', ' ', str(para[sent]))
        sent_text = re.sub('\\"', ' ', sent_text)
        # Substituting multiple spaces with single space
        sent_text = re.sub(r'\s+', ' ', sent_text, flags=re.I)
        sent_text = re.sub('\\n', ' ', sent_text)
        sent_text = re.sub('[^A-Za-z0-9]+', ' ', sent_text)
        # Remove stopwords and ensure all text is in lowercase.
        sent_text = ' '.join(e for e in sent_text.split() if e not in stopwords.words('english'))
        preprocessed_text.append(sent_text.lower())
    return preprocessed_text

#Function for computing semantic similarity between 2 text paragraphs
def sem_score(paragraph1, paragraph2):
    #Create a DataFrame for 2 paragraphs as columns
    df = pd.DataFrame({
        "Paragraph1": [paragraph1],
        "Paragraph2": [paragraph2]
    })
    
    #Vector representation of the paragraph and returns avergae of token embeddings 
    def embedding(paragraph):
        # If the paragraph is a string, split it into sentences.
        sentences = sent_tokenize(paragraph) if isinstance(paragraph, str) else paragraph
        # Preprocess all sentences with above preprocess_text function
        preprocessed_sentences = preprocess_text(sentences)
        tokens = []
        # Tokenize each preprocessed sentence
        for sent in preprocessed_sentences:
            tokens.extend(sent.split())
        
        # Generate embeddings for all tokens 
        token_vecs = [model[word] for word in tokens if word in model]
        
        #Return a zero vector if no tokens exist
        if not token_vecs:
            return np.zeros(model.vector_size)
        #Else return the average of the tokenized vectors
        return np.mean(token_vecs, axis=0)
    
    #Vector conversion for text paragraphs
    for idx, row in df.iterrows():
        emb1 = embedding(row["Paragraph1"])
        emb2 = embedding(row["Paragraph2"])
        
        # Calculates cosine similarity between embeddings
        sim = 1 - cosine(emb1, emb2)
    return sim

st.title("Semantic Text Similarity Score")
sentence1 = st.text_input("Enter the first sentence:")
sentence2 = st.text_input("Enter the second sentence:")

# User inputs for streamlit app
#sentence1 = st.text_input("Enter the first sentence:")
st.json(
    {
        "text1": sentence1,
        "text2": sentence2,
    },
)
#Button for computing similarity score of above input sentences
if st.button("Compute Similarity Score"):
    if sentence1 and sentence2:
        word2vec_similarity = sem_score(sentence1, sentence2)
        #st.write({f"\"similarity score\": {word2vec_similarity:.2f}"})
        st.write({"similarity score": + word2vec_similarity})