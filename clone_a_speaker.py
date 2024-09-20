import torch
import numpy as np
import json
import os
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager
from pydantic import BaseModel



# Assuming the TTS model and other necessary setup are done here
# Initialize and load your TTS model
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.device == "cuda":
        torch.cuda.empty_cache()
    else:
        print("WARNING: USING CPU")

    # Load or download your model
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"  # Example model name
    model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))

    if not os.path.exists(model_path):
        print("Downloading XTTS Model:", model_name)
        ModelManager().download_model(model_name)
        print("XTTS Model downloaded")

    print("Loading XTTS")
    config = XttsConfig()
    config.load_json(os.path.join(model_path,"config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_path, eval=True, use_deepspeed=False)
    model.to(device)
    print("XTTS Loaded.")

    return model
def predict_speaker_and_save(file_paths: list, model):
    """Compute conditioning inputs from multiple audio files and save them to a single file."""
    # Verify all files exist
    for file_path in file_paths:
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"

    # Process the list of files to generate a single embedding
    with torch.inference_mode():
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(file_paths)

    # Prepare data to be saved
    data_to_save = {
        "gpt_cond_latent": gpt_cond_latent.cpu().squeeze().half().tolist(),
        "speaker_embedding": speaker_embedding.cpu().squeeze().half().tolist(),
    }

    # Save the data to a file
    filename = "normal_girl.json"
    with open(filename, 'w') as file:
        json.dump(data_to_save, file)

    return f"Combined speaker embedding saved to {filename}"



# Example usage
file_path = ['how much youtube paid me for a year as a full time youtuber  my analytics with 50K subscribers.wav']
model = load_model()
result = predict_speaker_and_save(file_path, model)
print(result)
