import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.manage import ModelManager
import scipy.io.wavfile as wavfile
import json
import os
import numpy as np
import soundfile as sf
import re

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Use the default model name
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"

    # Download (if needed) and load the model
    model_manager = ModelManager()
    model_path, config_path, model_item = model_manager.download_model(model_name)
    config_path = os.path.join(model_path, "config.json")

    print("Loading XTTS")
    config = XttsConfig()
    config.load_json(file_name=config_path)  # Load from the downloaded config
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_path, eval=True, use_deepspeed=False)
    print("XTTS Loaded.")
    return model

def load_speaker_embedding(file_path):
    """Load speaker embedding from a file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['speaker_embedding'], data['gpt_cond_latent']

def split_text(text, max_tokens, tokenizer, language):
    """Split text into chunks that are within the max token limit."""
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Encode the current chunk plus the next sentence
        tokens = tokenizer.encode(current_chunk + " " + sentence, lang="en")
        if len(tokens) <= max_tokens:
            current_chunk = (current_chunk + " " + sentence).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
            # If the sentence itself exceeds max_tokens, we need to split it further
            tokens = tokenizer.encode(current_chunk, lang="en")
            if len(tokens) > max_tokens:
                # Split the sentence into smaller parts (e.g., words)
                words = current_chunk.split()
                sub_chunk = ""
                for word in words:
                    tokens = tokenizer.encode(sub_chunk + " " + word, lang="en")
                    if len(tokens) <= max_tokens:
                        sub_chunk = (sub_chunk + " " + word).strip()
                    else:
                        if sub_chunk:
                            chunks.append(sub_chunk)
                        sub_chunk = word
                if sub_chunk:
                    chunks.append(sub_chunk)
                current_chunk = ""
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def synthesize_speech(text_chunks, language, speaker_embedding, gpt_cond_latent, model):
    """Generate speech from text chunks using the specified speaker embedding."""
    # Convert to tensors
    speaker_embedding_tensor = torch.tensor(speaker_embedding).unsqueeze(0).unsqueeze(-1).to(model.device)
    gpt_cond_latent_tensor = torch.tensor(gpt_cond_latent).reshape((-1, 1024)).unsqueeze(0).to(model.device)

    # Initialize an empty list to hold audio outputs
    audio_outputs = []

    for idx, text in enumerate(text_chunks):
        print(f"Synthesizing chunk {idx+1}/{len(text_chunks)}")
        # Synthesize speech for each chunk
        out = model.inference(
            text,
            language,
            gpt_cond_latent_tensor,
            speaker_embedding_tensor,
            split_sentences=False,  # Already split sentences
        )
        if "wav" not in out or len(out["wav"]) == 0:
            raise ValueError(f"No audio data generated from the model for chunk {idx+1}!")
        audio_outputs.append(out["wav"])

    # Concatenate all audio outputs
    full_audio = np.concatenate(audio_outputs)

    return full_audio

def postprocess(wav):
    """Post-process the output waveform."""
    # Ensure `wav` is a numpy array
    wav = np.asarray(wav)

    # Normalize: Ensure the audio is between -1 and 1
    wav = np.clip(wav, -1.0, 1.0)

    # Scale to int16 range (-32768 to 32767) and convert to integer
    wav = (wav * 32767).astype(np.int16)

    return wav

if __name__ == "__main__":
    model = load_model()
    speaker_embedding, gpt_cond_latent = load_speaker_embedding("normal_girl.json")

    text = ''' 
    My(14M) brother's(17M) girlfriend(17F) came over for dinner at our house tonight. My parents are from Taiwan and at home we normally eat with chopsticks. This is my first time meeting my brother's girlfriend; she's white, and I wasn't trying to be rude or anything, but when I was setting the table, I just handed her training chopsticks. She looks at me confused and then says thank you. I continue to set the table like nothing is wrong. We all finally sit down to eat, and as we are about to eat, my sister(19F) asked my brother's girlfriend if she used chopsticks before and if she needed a fork. My brother's girlfriend said, "I'm actually pretty good with chopsticks! I was just given training ones for some reason," and when the entire room all at once looks at me—I truly mean ALL AT ONCE—I then say, "What? It was a logical assumption." My mom gets up and gets her regular chopsticks, and after dinner, my mom told me I'm embarrassing and she probably thinks we hate her now.
    '''

    # Get the tokenizer from the model
    tokenizer = model.tokenizer
    max_tokens = 400  # Maximum tokens allowed by the model

    # Split the text into manageable chunks
    text_chunks = split_text(text, max_tokens, tokenizer, language="en")

    print(f"Total chunks to synthesize: {len(text_chunks)}")

    # Synthesize speech
    full_wav = synthesize_speech(
        text_chunks,
        "en",
        speaker_embedding,
        gpt_cond_latent,
        model
    )

    # Post-process and convert to audio format
    wav = postprocess(full_wav)

    # Save the audio file
    sf.write('output.wav', wav, 22050, 'PCM_16')

    print("Speech synthesized and saved to output.wav")
