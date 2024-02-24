import streamlit as st
import mne
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from colorama import Fore, Style
import tempfile
import os
from collections import OrderedDict
import librosa


# Define the channel pairs and their joined names
channel_pairs = [
    ['EEG Fp1', 'EEG F7'], ['EEG F7', 'EEG T3'], ['EEG T3', 'EEG T5'], ['EEG T5', 'EEG O1'],
    ['EEG Fp1', 'EEG F3'], ['EEG C3', 'EEG F3'], ['EEG F3', 'EEG O1'], ['EEG Fp2', 'EEG F4'],
    ['EEG F4', 'EEG C4'], ['EEG C4', 'EEG P4'], ['EEG P4', 'EEG O2'], ['EEG Fp2', 'EEG F8'],
    ['EEG F8', 'EEG T4'], ['EEG T4', 'EEG T6'], ['EEG T6', 'EEG O2']
]
channel_pairs_joined = ['{}-{}'.format(pair[0], pair[1]) for pair in channel_pairs]

# Load the pre-trained machine learning model
model_file = 'epilepsy_prediction_model.pkl'
model = joblib.load(model_file)

# Define the target sampling rate
target_sampling_rate = 512  # in Hz

# Function to print decorative log
def print_decorative_log(message, color=Fore.BLUE, style=Style.RESET_ALL):
    line_length = len(message) + 4  # Length of the message plus padding on both sides
    decorative_line = "#" * line_length
    print(color + decorative_line)
    print(f"# {message} #")
    print(decorative_line + style)

# Function to compute cepstrum_mel
def compute_cepstrum_mel(data, sfreq, n_mfcc=20):
    mfccs = librosa.feature.mfcc(y=data, sr=sfreq, n_mfcc=n_mfcc)
    return mfccs

# Function to preprocess the raw data
def preprocess_raw(raw):
    # Preprocessing steps...
    # Preprocessing steps...

    print_decorative_log("Starting Preprocessing Sequence", Fore.GREEN)
               
    # Select the desired channels from channel pairs which resemble the bipolar longitudinal  channels of 10-20 system 
    selected_channels = []
    [selected_channels.extend(pair) for pair in channel_pairs if pair not in selected_channels]

    selected_channels = list(OrderedDict.fromkeys(selected_channels))
    selected_channels.append('2')

    #Drop extra channels
    # Check the number of channels
    #if len(raw.ch_names) > 35:
    for i, channel_name in enumerate(raw.ch_names):
        if 'EEG FP2' in channel_name:
            raw.rename_channels({channel_name: 'EEG Fp2'})
    # Drop channels not found in the desired channel list
    channels_to_drop = [channel_name for channel_name in raw.ch_names if channel_name not in selected_channels]
    raw.drop_channels(channels_to_drop)
    print_decorative_log("Extra Channels Dropped ... ", Fore.RED)

    # Reorder the channels to match the standard ordering for the dataset
    channels_order = selected_channels
    # Reorder channels
    raw = raw.pick(channels_order)
    print_decorative_log("Channels Reordered ... ", Fore.YELLOW)       
    # Set the channel type for '2' to 'ecg'
    raw.set_channel_types({'2': 'ecg'})
    
    print_decorative_log("ECG Channel Selected ... ", Fore.YELLOW)

    # Filtering to remove slow drifts
    filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None)
    print_decorative_log("Slow drifts removed ... ", Fore.YELLOW)

    # Apply ICA to remove ECG artifacts

    ica = mne.preprocessing.ICA(n_components=15, max_iter="auto", random_state=97)
    ica.fit(filt_raw)
    ica.exclude = []
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw, method="correlation", threshold="auto")
    ica.exclude = ecg_indices

    reconst_raw = raw.copy()
    ica.apply(reconst_raw)

    print_decorative_log("ECG Artificats Removed... ", Fore.YELLOW)


    # Perform bipolar longitudinal referencing
    anodes = []
    cathodes = []
    for pair in channel_pairs:
        anodes.append(pair[0])
        cathodes.append(pair[1])

    raw_bip_ref = mne.set_bipolar_reference(reconst_raw, anode=anodes, cathode=cathodes)
    raw_bip_ref_ch = raw_bip_ref.copy().pick_channels(channel_pairs_joined)
    print_decorative_log("Bipolar Referencing Done ... ", Fore.YELLOW)
    raw_clean = mne.preprocessing.oversampled_temporal_projection(raw_bip_ref_ch)
    raw_clean.filter(0.0, 40.0)
    print_decorative_log("Smoothing & Filtering Done ... ", Fore.YELLOW)

    return raw_clean

# Function to simulate streaming data and make predictions
def simulate_streaming_data(raw, start_time, end_time):
    st.write("Starting Simulation")

    # Crop the raw data to the specified start and end time
    raw.crop(tmin=start_time, tmax=end_time)
    
    # Preprocess the raw data
    preprocessed_raw = preprocess_raw(raw)

    # Get the data and the corresponding time vector
    data = preprocessed_raw.get_data(picks=channel_pairs_joined)#, tmin=1138, tmax=1218)
    time = preprocessed_raw.times

    # Define the window size for frame sampling
    window_size = 2 # Window size in seconds

    # Calculate the number of samples in the window
    window_samples = int(window_size * target_sampling_rate)

    # Calculate the number of frames
    num_frames = int(len(data[0]) / window_samples)
    print(num_frames)

    # Iterate over the frames
    for frame_idx in range(num_frames):
        # Calculate the start and end sample indices for the current frame
        start_idx = frame_idx * window_samples
        end_idx = start_idx + window_samples

        # Extract the frame data for all channels
        frame_data = data[:, start_idx:end_idx]

        # Compute mfccs
        n_mfcc = 20  # Number of MFCC coefficients
        cepstrum_mel_features = []
        for channel_data in frame_data:
            cepstrum_mel = compute_cepstrum_mel(channel_data, target_sampling_rate, n_mfcc)
            cepstrum_mel_features.append(cepstrum_mel)
        cepstral_features = np.concatenate(cepstrum_mel_features, axis=0)

        # Convert features to DataFrame
        frame_df = pd.DataFrame(cepstral_features.T)

        # Apply feature scaling to the latest frame data
        scaler = StandardScaler()

        frame_scaled = scaler.fit_transform(frame_df)

        # Make prediction using the pre-trained model
        prediction = model.predict(frame_scaled)[0]

        # Map the predicted label to the corresponding class
        class_mapping = {0: 'pre-ictal', 1: 'ictal', 2: 'post-ictal', 3: 'normal'}
        predicted_class = class_mapping[prediction]

        # Display the streaming data and classification result
        st.subheader("Streaming 10 secs")
        st.info(f"Classification Result: {predicted_class}")
        st.write("--------------------------------")

# Streamlit app
@st.cache(allow_output_mutation=True)
def load_data(file_path):
    raw = mne.io.read_raw_edf(file_path)
    return raw

def main():
    st.title("EDF Streaming Data Classification")

    # File upload and user input
    uploaded_file = st.file_uploader("Upload EDF file", type=["edf"])

    if uploaded_file is not None:
            # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
            tmp_filename = tmp_file.name
            tmp_file.write(uploaded_file.read())
            print(tmp_filename)
        # Read the EDF file using mne.io.read_raw
        raw = mne.io.read_raw_edf(tmp_filename, preload=True)
        # Perform further processing or analysis with the raw data

        # Remove the temporary file
        os.remove(tmp_filename)

        start_time = st.number_input("Start Time (in seconds)", min_value=0.0, max_value=raw.times[-1], value=0.0)
        end_time = st.number_input("End Time (in seconds)", min_value=start_time, max_value=raw.times[-1], value=raw.times[-1])

        if st.button("Start Classification"):
            simulate_streaming_data(raw, start_time, end_time)

if __name__ == "__main__":
    main()