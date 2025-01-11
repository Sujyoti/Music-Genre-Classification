import streamlit as st
import tensorflow as tf
import numpy as np
import os
import librosa
from matplotlib import pyplot
from tensorflow.image import resize  # type: ignore
from PIL import Image  


#Function
@st.cache_resource()
def load_model():
  model = tf.keras.models.load_model("Trained_model.h5")
  return model


# Load and preprocess audio data
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    # Perform preprocessing (e.g., convert to Mel spectrogram and resize)
    # Define the duration of each chunk and overlap
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds
                
    # Convert durations to samples
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
                
    # Calculate the number of chunks
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
                
    # Iterate over each chunk
    for i in range(num_chunks):
                    # Calculate start and end indices of the chunk
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
                    
                    # Extract the chunk of audio
        chunk = audio_data[start:end]
                    
                    # Compute the Mel spectrogram for the chunk
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
                    
                #mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)
    
    return np.array(data)



#Tensorflow Model Prediction
def model_prediction(X_test):
    model = load_model()
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred,axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    #print(unique_elements, counts)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return max_elements[0]



#sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About Project","Prediction"])

## Main Page
if(app_mode=="Home"):
    st.markdown(
    """
    <style>
    .stApp {
        background-color: #131010;  /* Blue background */
        color: white;
    }
    h2, h3 {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

    st.markdown(''' ## Welcome to the,\n
    ## Music Genre Classification System! 🎶🎧''')
    image_path = "photo.png"
    st.image(image_path, use_column_width=True)
    st.markdown("""
**Goal is to help in identifying music genres from audio tracks efficiently. Upload an audio file, and our system will analyze it to detect its genre. Discover the power of AI in music analysis!**

### How It Works
1. **Upload Audio:** Go to the **Genre Classification** page and upload an audio file.
2. **Analysis:** Our system will process the audio using advanced algorithms to classify it into one of the predefined genres.
3. **Results:** View the predicted genre along with related information.

### Why Choose Us?
- **Accuracy:** Our system leverages state-of-the-art deep learning models for accurate genre prediction.
- **User-Friendly:** Simple and intuitive interface for a smooth user experience.
- **Fast and Efficient:** Get results quickly, enabling faster music categorization and exploration.

### Get Started
Click on the **Genre Classification** page in the sidebar to upload an audio file and explore the magic of our Music Genre Classification System!

### About Us
Learn more about the project, our team, and our mission on the **About** page.
""")



#About Project
elif(app_mode=="About Project"):
    st.markdown("""
                ### About Project
                Music Experts have been trying for a long time to understand sound and what differenciates one song from another. How to visualize sound. What makes a tone different from another.

             
                ### About Dataset
               
                The GTZAN dataset was introduced by George Tzanetakis in 2002 for music genre classification and is often used as a benchmark dataset in research. The dataset was used in more than 100 papers already in 2013 according to a survey. It is popular since the concept of music genres and single-label classification is easy, simple, and straightforward.
                
                Total Files : 1000 audio tracks.

                Genres : 10 music genres, each with 100 tracks: 
                (Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock).

                Duration : Each track is 30 seconds long.

                Format : WAV files with a sampling rate of 22.05 kHz and mono audio.

                """)

    

#Prediction Page
elif(app_mode=="Prediction"):
    st.header("Model Prediction")
    test_mp3 = st.file_uploader("Upload an audio file", type=["mp3"])
    if test_mp3 is not None:
           ## filepath = "C:\Users\sujyo\OneDrive\Desktop\Music Genre Classification-20241218T170735Z-001\Music Genre Classification\Music" + test_mp3.name
            filepath = "./Music/" + test_mp3.name


            ##filepath = os.path.join(os.getcwd(), 'Music', 'Musickpop-128609.mp3')
          

    #Show Button
    if(st.button("Play Audio")):
        st.audio(test_mp3)
    
    #Predict Button
    if(st.button("Predict")):
      with st.spinner("Please Wait..."):       
        X_test = load_and_preprocess_data(filepath)
        result_index = model_prediction(X_test)
        st.balloons()
        label = ['blues', 'classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
        st.markdown("**:blue[Model Prediction:] It's a  :red[{}] music**".format(label[result_index]))

       