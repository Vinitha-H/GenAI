import os

import streamlit as st
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)

import base64
import vertexai.preview.generative_models as generative_models
import google.cloud.texttospeech as tts

PROJECT_ID = 'genai-420915' # os.environ.get("GCP_PROJECT")  # Your Google Cloud Project ID
LOCATION = 'us-central1' # os.environ.get("GCP_REGION")  # Your Google Cloud Project Region
vertexai.init(project=PROJECT_ID, location=LOCATION)

# print(PROJECT_ID, LOCATION)

st.set_page_config(page_title="Translate something")

@st.cache_resource
def load_models():
    """
    Load the generative models for text and multimodal generation.

    Returns:
        Tuple: A tuple containing the text model and multimodal model.
    """
    text_model_pro = GenerativeModel("gemini-1.0-pro-vision")
    multimodal_model_pro = GenerativeModel("gemini-1.5-pro-preview-0409")
    return text_model_pro, multimodal_model_pro

def get_gemini_pro_text_response(
    model: GenerativeModel,
    contents: str,
    generation_config: GenerationConfig,
    stream: bool = True,
):
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    responses = model.generate_content(
        prompt,
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=stream,
    )

    final_response = []
    for response in responses:
        try:
            # st.write(response.text)
            final_response.append(response.text)
        except IndexError:
            # st.write(response)
            final_response.append("")
            continue
    return " ".join(final_response)


def get_gemini_pro_vision_response(
    model, prompt_list, generation_config={}, stream: bool = True
):
    # generation_config = {"temperature": 0.1, "max_output_tokens": 2048}
    responses = model.generate_content(
        prompt_list, generation_config=generation_config, stream=stream
    )
    final_response = []
    for response in responses:
        try:
            final_response.append(response.text)
        except IndexError:
            pass
    return "".join(final_response)

st.header("Vertex AI Gemini", divider="rainbow")
text_model_pro, multimodal_model_pro = load_models()

def text_to_wav(voice_name: str, text: str):
    language_code = "-".join(voice_name.split("-")[:2])
    text_input = tts.SynthesisInput(text=text)
    voice_params = tts.VoiceSelectionParams(
        language_code=language_code, name=voice_name
    )
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16)

    client = tts.TextToSpeechClient()
    response = client.synthesize_speech(
        input=text_input,
        voice=voice_params,
        audio_config=audio_config,
    )

    filename = f"{voice_name}.wav"
    print("sound file:", filename)
    with open(filename, "wb") as out:
        out.write(response.audio_content)
        print(f'Generated speech saved to "{filename}"')

    if os.path.isfile(f"./{voice_name}.wav"):
        st.audio(f"./{voice_name}.wav", format="audio/wav", loop=False)

with st.sidebar:
    st.title('Gemini Bot')
    st.subheader('Models and parameters')
    selected_model = "gemini-1.5-pro-preview-0409"

    # Premise
    from_language = st.radio(
        "Select the language: \n\n",
        ["English", "German", "French", "Italian"],
        key="from_language",
        horizontal=True,
    )

    to_language = st.radio(
        "Select the language: \n\n",
        ["German", "English", "French", "Italian"],
        key="to_language",
        horizontal=True,
    )
    st.markdown(f'Learn to speak {to_language}')
    
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=1.0, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.95, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=2048, max_value=8192, value=2048, step=8)

st.write("Using Gemini 1.0 Pro - Text only model")
st.subheader("Translate a scenario")

st.markdown(f'Learn to speak {to_language}')
task = st.text_input(
    "Task: \n\n", key="task", placeholder="Type a sentence"
)
task_response = ''
if task:
    config = {
    "temperature": 0.8,
    "max_output_tokens": 2048,
    }
    prompt1 = f"""Mention the language in which the text within quotes "{task}" is written in, in just one word"""
    task_response = get_gemini_pro_vision_response( # changed from text_response
                text_model_pro,
                prompt1,
                generation_config=config,
            )
    if task_response:
        st.write("The text is in:")
        st.write(task_response)


if task_response!= '' and from_language != task_response.strip():
    from_language = task_response

prompt1 = f"""I want to learn to form grammatically correct sentences in {to_language} based on a scenario. You need to assist with translating the sentence within quotes "{task}" to {to_language}, otherwise, if the sentence is in {to_language}, verify if the sentence is grammatically correct in {to_language} and and if is not grammatically correct, provide the corrected sentence in {to_language} and explain the reason why the original sentence is not correct along with the correct grammar. Additionally provide alternative sentence constructs with similar meaning."""
config = {
    "temperature": temperature, # 0.8,
    "max_output_tokens": 2048,
}

generate_t2t = st.button("Check my sentence", key="generate_t2t")
if generate_t2t and prompt1:
    # st.write(prompt1)
    with st.spinner("Checking your sentence using Gemini 1.0 Pro ..."):
        first_tab1, first_tab2 = st.tabs(["Translation", "Prompt"])
        with first_tab1:
            response = get_gemini_pro_vision_response( # changed from text_response
                text_model_pro,
                prompt1,
                generation_config=config,
            )
            if response:
                st.write("Your translation:")
                st.write(response)

            # generate_audio = st.button("Generate Audio", key="generate_audio", on_click=text_to_wav, args=["de-DE-Neural2-B", response])
            # st.markdown(response)
            match to_language:
                case "English":
                    speech_lang = "en-GB-Neural2-B"
                case "German":
                    speech_lang = "de-DE-Neural2-B"
                case "French":
                    speech_lang = "fr-FR-Neural2-A"
                case "Italian":
                    speech_lang = "it-IT-Neural2-A"
                case _:
                    speech_lang = "en-GB-Neural2-B"
            text_to_wav(speech_lang, response.replace('*', ''))
            # if os.path.isfile('./de-DE-Neural2-B.wav'):
            #     st.audio("de-DE-Neural2-B.wav", format="audio/wav", loop=False)

            # if generate_audio:
            #     print('generating audio')
            #     text_to_wav("de-DE-Neural2-B", response)

        with first_tab2:
            st.markdown(prompt1)




