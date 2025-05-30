import streamlit as st
import numpy as np
import io
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine
from gensim.models import KeyedVectors

# Download NLTK data if not already present
#nltk.download('punkt')
#nltk.download('stopwords')

#Function for performing text pre=processing
def preprocess_text(para):
    preprocessed_text = []
    for sent in range(len(para)):
        # Remove newline characters, quotes, spaces and non-alphanumeric characters
        sent_text = re.sub('\\r', ' ', str(para[sent]))
        sent_text = re.sub('\\"', ' ', sent_text)
        sent_text = re.sub(r'\s+', ' ', sent_text, flags=re.I)
        sent_text = re.sub('\\n', ' ', sent_text)
        sent_text = re.sub('[^A-Za-z0-9]+', ' ', sent_text)
        # Remove stopwords and convert them to lower case
        sent_text = ' '.join(e for e in sent_text.split() if e not in stopwords.words('english'))
        preprocessed_text.append(sent_text.lower())
    return preprocessed_text

#Function for computing semantic similarity between 2 text paragraphs
def sem_score(paragraph1, paragraph2):
    #Vector representation of the paragraph and returns average of token embeddings
    def embedding(paragraph):
        #If paragraph is a string, split it into sentences
        sentences = sent_tokenize(paragraph) if isinstance(paragraph, str) else paragraph
        #Preprocess all sentences with above preprocess_text function
        processed_sentences = preprocess_text(sentences)
        tokens = []
        #Tokenize each preprocessed sentence
        for sent in processed_sentences:
            tokens.extend(sent.split())
        #Generate embeddings for all tokens found in the model
        token_vecs = [st.session_state.model[word] for word in tokens if word in st.session_state.model]
        #Returns a zero vector if no tokens exist
        if not token_vecs:
            return np.zeros(st.session_state.model.vector_size)
        return np.mean(token_vecs, axis=0)
    
    emb1 = embedding(paragraph1)
    emb2 = embedding(paragraph2)
    # Calculates cosine similarity between embeddings
    sim = 1 - cosine(emb1, emb2)
    return sim


#Streamlit app
st.title("Sentence Similarity Checker")

#Use session_state to check whether the model has been loaded
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

#Displays the file uploader widget only if the model has not been loaded
if not st.session_state.model_loaded:
    uploaded_file = st.file_uploader("Upload your pre-trained Word2Vec model file (.bin or .gz)", type=["bin", "gz"])
    if uploaded_file is not None:
        #Read the uploaded file into bytes
        file_bytes = uploaded_file.read()
        #Store the file in a NumPy buffer.
        buffer_array = np.frombuffer(file_bytes, dtype=np.uint8)
        bytes_data = buffer_array.tobytes()
        
        #Creates a BytesIO object, acting like a file
        model_file = io.BytesIO(bytes_data)
        model_file.seek(0)
        
        #Load the model from the file-like object
        st.session_state.model = KeyedVectors.load_word2vec_format(model_file, binary=True, limit=500000)
        st.session_state.model_loaded = True
        
        
#Hides the uploader and show paragraph inputs after model is loaded
if st.session_state.model_loaded:
    st.write("Enter two paragraphs to compute their semantic similarity:")
    paragraph1 = st.text_area("Enter the first paragraph:")
    paragraph2 = st.text_area("Enter the second paragraph:")
    
    st.json(
    {
        "text1": paragraph1,
        "text2": paragraph2,
    },)
    
    #Button for computing similarity score of above input text paras
if st.button("Compute Similarity Score"):
    if paragraph1 and paragraph2:
        word2vec_similarity = sem_score(paragraph1, paragraph2)
        #st.write({f"\"similarity score\": {word2vec_similarity:.2f}"})
        st.write({"similarity score": + word2vec_similarity})