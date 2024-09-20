import os
import pandas as pd
import numpy as np
import json
import re
import soundfile as sf
import torch
import tempfile
import wave
from tqdm import tqdm
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager


def postprocess(wav):
    """Post process the output waveform"""
    if isinstance(wav, list):
        wav = torch.cat(wav, dim=0)
    wav = wav.clone().detach().cpu().numpy()
    wav = wav[None, :int(wav.shape[0])]
    wav = np.clip(wav, -1, 1)
    wav = (wav * 32767).astype(np.int16)
    return wav
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        

    # Load or download your model
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"  # Example model name
    model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))

    if not os.path.exists(model_path):
        print("Downloading XTTS Model:", model_name)
        ModelManager().download_model(model_name)
        print("XTTS Model downloaded")

    print("Loading XTTS")
    config = XttsConfig()
    config_file = os.path.join(model_path, "config.json")  # Use the default config file
    config.load_json(config_file)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_path, eval=True, use_deepspeed=False)
    model.to(device)
    print("XTTS Loaded.")

    return model

def load_speaker_embedding(file_path):
    """Load speaker embedding from a file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['speaker_embedding'], data['gpt_cond_latent']

def synthesize_speech(text, language, speaker_embedding, gpt_cond_latent, model):
    """Generate speech from text using the specified speaker embedding."""
    # Convert to tensors
    speaker_embedding_tensor = torch.tensor(speaker_embedding).unsqueeze(0).unsqueeze(-1)
    gpt_cond_latent_tensor = torch.tensor(gpt_cond_latent).reshape((-1, 1024)).unsqueeze(0)

    # Synthesize speech
    out = model.inference(
        text,
        language,
        gpt_cond_latent_tensor,
        speaker_embedding_tensor,
    )

    # Post-process and convert to audio format
    wav = postprocess(torch.tensor(out['wav']))

    # Save or return the audio data
    return wav
def get_wav_duration(file_path):
    """Calculate the duration of a WAV file in seconds."""
    with wave.open(file_path, 'rb') as wav_file:
        framerate = wav_file.getframerate()
        nframes = wav_file.getnframes()

        # Calculate duration
        duration = nframes / float(framerate)
        return duration

def split_sentence(sentence, max_length=250):
    """
    Split a sentence at the nearest comma to the midpoint for sentences longer than max_length.
    If no comma is found, split at the nearest space.
    """
    if len(sentence) <= max_length:
        return [sentence]

    # Calculate the midpoint of the sentence
    midpoint = len(sentence) // 2

    # Find the nearest comma to the midpoint
    left_comma = sentence.rfind(',', 0, midpoint)
    right_comma = sentence.find(',', midpoint)

    # Choose the closest comma to split
    if left_comma != -1 or right_comma != -1:
        # Prioritize splitting at a comma
        if right_comma == -1 or (left_comma != -1 and midpoint - left_comma <= right_comma - midpoint):
            split_index = left_comma
        else:
            split_index = right_comma
    else:
        # If no comma found, find the nearest space
        left_space = sentence.rfind(' ', 0, midpoint)
        right_space = sentence.find(' ', midpoint)
        if right_space == -1 or (left_space != -1 and midpoint - left_space <= right_space - midpoint):
            split_index = left_space
        else:
            split_index = right_space

    # Split the sentence
    first_part = sentence[:split_index + 1].rstrip()  # Include the comma in the first part
    second_part = sentence[split_index + 1:].lstrip()

    # # Debug print
    print(f"Original Sentence: '{sentence}'")
    print(f"Split Index: {split_index}")
    print(f"First Part: '{first_part}'")
    print(f"Second Part: '{second_part}'")

    return [first_part, second_part]

def split_text_into_sentences(text):
    """Split the text into sentences using punctuation as a delimiter."""
    return re.split(r'(?<=[.!?]) +|\n+', text.strip())

def generate_silence(duration_ms, sample_rate):
    """Generate a silence (zero amplitude) segment of a given duration in milliseconds."""
    num_samples = int(sample_rate * duration_ms / 1000)
    return np.zeros(num_samples, dtype=np.int16)


def create_srt(subtitles, output_file_path):
    """ Create an SRT file for the given subtitles with timings. """
    with open(output_file_path, "w", encoding='utf-8') as file:
        for index, (start, end, text) in enumerate(subtitles, 1):
            start_time_str = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{int(start%60):02d},000"
            end_time_str = f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{int(end%60):02d},000"
            file.write(f"{index}\n")
            file.write(f"{start_time_str} --> {end_time_str}\n")
            file.write(f"{text.strip()}\n")
            file.write("\n")  # Only one newline to separate entries

def process_sentences(sentences, model, language, speaker_embedding, gpt_cond_latent, sample_rate, sample_width=2):
    full_wav_data = np.array([], dtype=np.int16)
    # Load the speaker embedding once
    subtitles = []
    start_time = 0.0
    pause_duration_ms = 500  # Adjust this value as needed
    for sentence in sentences:
        parts = split_sentence(sentence)
        subtitle_start = start_time
        for i, part in enumerate(parts):
            wav_data = synthesize_speech(part, language, speaker_embedding, gpt_cond_latent, model)
            part_end_index = sentence.find(part) + len(part)
            if i < len(parts) - 1 and sentence[part_end_index:part_end_index + 1] == ',':
                silence = generate_silence(pause_duration_ms, sample_rate)
                wav_data = np.concatenate((wav_data, silence))
            wav_data = np.asarray(wav_data, dtype=np.int16).flatten()

            # Write to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                # Save the audio data to the temporary file
                sf.write(temp_file.name, wav_data, sample_rate, subtype='PCM_16')
                # Calculate duration using the temporary file
                duration = get_wav_duration(temp_file.name)

            # Append to full WAV data
            full_wav_data = np.concatenate((full_wav_data, wav_data))

            # Update end time for this part
            end_time = start_time + duration

            # Update start time for next part
            start_time = end_time

            # Clean up the temporary file
            os.remove(temp_file.name)

        # Generate and append silence after processing the sentence
        silence = generate_silence(pause_duration_ms, sample_rate)
        full_wav_data = np.concatenate((full_wav_data, silence))

        # Update start_time for the next sentence, accounting for the pause
        start_time += pause_duration_ms / 1000.0
        # Append subtitle for the whole sentence
        subtitles.append((subtitle_start, end_time, sentence))

        # Update the progress bar
        pbar.update(1)

    return full_wav_data, subtitles

if __name__ == "__main__":

    
    # Process sentences and create subtitles
    model = load_model()
  
    # Load the text from a file
    with open("text_example_preprocessed.txt", "r") as file:
        text = file.read()


    # Split the text into sentences
    sentences = split_text_into_sentences(text)

    language = "en"
    speaker = "audiobook_lady.json"

    pbar = tqdm(total=len(sentences), desc="Processing Text")


    # Set the sample rate and sample width
    sample_rate = 24000  # 24 kHz sampling rate
    sample_width = 2     # 16 bits (2 bytes)

    # Load the speaker embedding and GPT condition latent once
    speaker_embedding, gpt_cond_latent = load_speaker_embedding(f"speakers/{speaker}")

    # Call process_sentences with the loaded embeddings
    full_wav_data, subtitles = process_sentences(sentences, model, language, speaker_embedding, gpt_cond_latent, sample_rate, sample_width)

    file_name = "output"
    
    pbar.close()

    # Specify the output file path for the synthesized speech
    output_wav_file_path = f'output/{file_name}.wav'
    subtitles_output_path = f'output/{file_name}.srt'

    create_srt(subtitles, subtitles_output_path)
    # Save the full WAV file
    sf.write(output_wav_file_path, full_wav_data, sample_rate, subtype='PCM_16')

    print(f"Synthesized speech saved to {output_wav_file_path}")
