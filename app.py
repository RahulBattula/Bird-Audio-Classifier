import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
from io import BytesIO
import plotly.graph_objects as go

# Load the trained models and label encoders
model_lstm = load_model("audio_classification_lstm.hdf5")
model_sequential = load_model("audio_classification_sequential.hdf5")  # Replace with your Sequential model path
# model_vggish = load_model("audio_classification_vggish.hdf5")  # Replace with your VGGish model path
model_gru = load_model("audio_classification_gru.hdf5")  # Replace with your VGGish model path

labelencoder_lstm = LabelEncoder()
labelencoder_lstm.classes_ = np.load("lstm_labelencoder.npy")

labelencoder_sequential = LabelEncoder()
labelencoder_sequential.classes_ = np.load("sequential_labelencoder.npy")  # Replace with your Sequential label encoder path

# labelencoder_vggish = LabelEncoder()
# labelencoder_vggish.classes_ = np.load("vggish_labelencoder.npy")  # Replace with your VGGish label encoder path

labelencoder_gru = LabelEncoder()
labelencoder_gru.classes_ = np.load("gru_labelencoder.npy")  # Replace with your VGGish label encoder path

# Set the browser tab icon (favicon)
st.set_page_config(
    page_title="Bird Audio Classifier",
    page_icon="🐦"  # You can use an emoji or provide a URL to an image
)

def predict_bird(audio_bytes, model, labelencoder):
    # Load and preprocess the audio file for prediction
    audio, sample_rate = librosa.load(BytesIO(audio_bytes), res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

    # Make prediction using the loaded model
    predicted_label = model.predict(mfccs_scaled_features)
    predicted_class_index = np.argmax(predicted_label)
    prediction_class = labelencoder.inverse_transform([predicted_class_index])

    return prediction_class[0]

# Set the title and description
st.title("Bird Audio Classifier")
st.write("Upload audio files and predict the bird!")

# Multifile uploader for audio files
uploaded_files = st.file_uploader("Choose audio files", type=["wav"], accept_multiple_files=True)

# Display the uploaded audio files, play button, and waveform with random colors
for i, uploaded_file in enumerate(uploaded_files):
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes, format='audio/wav', start_time=0)

    # Visualization - Waveform
    st.subheader(f"Waveform for {uploaded_file.name}")
    audio, _ = librosa.load(BytesIO(audio_bytes), sr=None)

    # Generate a random color
    color = f"rgb({int(np.random.rand() * 256)}, {int(np.random.rand() * 256)}, {int(np.random.rand() * 256)})"

    # Plot the waveform with the assigned color
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(audio)), y=audio, mode='lines', line=dict(color=color)))
    fig.update_layout(xaxis_title='Time (samples)', yaxis_title='Amplitude')
    st.plotly_chart(fig)

    # Button to predict the bird for each model
    if st.button(f"Predict for {uploaded_file.name}"):
        # Display loading message while predicting
        with st.spinner(f"Predicting for {uploaded_file.name}..."):
            # Predict using LSTM model
            predicted_bird_lstm = predict_bird(audio_bytes, model_lstm, labelencoder_lstm)
            st.success(f"For {uploaded_file.name}, LSTM predicted bird is: {predicted_bird_lstm}")

            # Predict using Sequential model
            predicted_bird_sequential = predict_bird(audio_bytes, model_sequential, labelencoder_sequential)
            st.success(f"For {uploaded_file.name}, Sequential predicted bird is: {predicted_bird_sequential}")

            # Predict using VGGish model
            # predicted_bird_vggish = predict_bird(audio_bytes, model_vggish, labelencoder_vggish)
            # st.success(f"For {uploaded_file.name}, VGGish predicted bird is: {predicted_bird_vggish}")

            # Predict using GRU model
            predicted_bird_gru = predict_bird(audio_bytes, model_gru, labelencoder_gru)
            st.success(f"For {uploaded_file.name}, GRU predicted bird is: {predicted_bird_gru}")

# Sidebar with app description and contact information
st.sidebar.title("About the App")
st.sidebar.write(
    "This web app allows you to upload bird audio files, predict the bird species, and visualize the waveform."
)

st.sidebar.title("Contact Me")
st.sidebar.write(
    "Feel free to contact me if you have any questions or suggestions. "
    "You can reach me at example@email.com."
)
