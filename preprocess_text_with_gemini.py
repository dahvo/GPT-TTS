import os
import google.generativeai as genai
from google.generativeai.types.generation_types import StopCandidateException
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv

load_dotenv()

import logging

logging.basicConfig(level=logging.INFO)

# Configure the API key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])


def preprocess_text(text):
    # Create the model
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-8b-exp-0827",
        generation_config=generation_config,
        # safety_settings = Adjust safety settings
        # See https://ai.google.dev/gemini-api/docs/safety-settings
        system_instruction="""You are a preprocessor of text for a TTS system. 
        You fix typos, and make all words phonetically simple to pronounce phonetically so that the TTS sounds clear and is easy to understand. 
        You will receive text that is intended to be read aloud by a TTS system. Your job is to rewrite the text to make it easier for the TTS system to pronounce, while preserving the original meaning as much as possible.

        **Here are some specific things you should do:**

        * **Expand contractions and abbreviations:** For example, change "I'm" to "I am" and "etc." to "etcetera".
        * **Rewrite words that are difficult to pronounce:** For example, change "February" to "Feb-roo-ary" and "Worcestershire" to "war-chester-shire".
        * **Remove or replace words that are likely to be mispronounced by the TTS system:** For example, change "colonel" to "kernel" and "quay" to "key".
        * **Break up long sentences into shorter ones:** This will make it easier for the TTS system to pause and breathe naturally.
        * **Add punctuation to clarify the meaning of the text:** This will help the TTS system to read the text with the correct intonation and emphasis.

        **You should avoid making any changes that would significantly alter the meaning of the original text.** For example, you should not change the order of words or add or remove any important information.

        **Your output should be text that is easy for a TTS system to pronounce and that conveys the same meaning as the original text.**
      """,
        safety_settings={
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        },
    )

    chat_session = model.start_chat(history=[])

    try:
        response = chat_session.send_message(text)
        return response.text

    except StopCandidateException as e:
        # Handle the safety
        print(f"StopCandidateException occurred: {e}")
        return None

    except Exception as e:
        # Handle any other exceptions
        print(f"An unexpected error occurred: {e}")
        return None
