# Text-to-Speech Processing

This repository contains a set of Python scripts for processing text-to-speech (TTS) tasks, including splitting sentences, generating silence, and creating subtitles.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/dahvo/GPT-TTS.git
    cd GPT-TTS
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Synthesize Speech
1. Create a new file in the home directory named main.py
```python
from do_tts import text_to_tts()

with open("text_example.txt" 
#Or whatever the file is called that holds the text you are converting to TTS. This is setup so ANY SIZE of text, even a book should work without a hitch
, "r", encoding="utf-8") as file:
        input_text = file.read()

    speaker = "audiobook_lady"
    #choose a speaker, either one from my examples or you may create your own speaker embedding using the provided files

    output_name = "test_output"
    #Name of the output for the audio and srt file

    text_to_tts(input_text, speaker, output_name)
    ```