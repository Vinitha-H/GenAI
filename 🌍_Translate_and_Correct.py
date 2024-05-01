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
st.config._set_option("global.disableWidgetStateDuplicationWarning", True, "test")

# print(PROJECT_ID, LOCATION)

st.set_page_config(page_title="LinguaCoach", page_icon="🌍")

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
        if response.text.strip() != '':
            try:
                # st.write(response.text)
                final_response.append(response.text)
            except IndexError:
                # pass
                final_response.append("")
                continue
            except Exception as e:
                print("The error is: ",e)
                pass
    return "".join(final_response)

st.header("LinguaCoach - Translate & Correct", divider="rainbow")
st.write("Lost in Translation? Not Anymore! Your Language & Grammar Wizard")
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
    st.title('LinguaCoach Bot')
    st.subheader('Parameters')
    selected_model = "gemini-1.5-pro-preview-0409"

    def update_lang():
        st.session_state.audio_language = st.session_state.to_language

    # Premise
    from_language = st.radio(
        "Select the 'source' language: \n\n",
        ["English", "German", "French", "Italian"],
        key="from_language",
        horizontal=True,
    )
    orig_lang = from_language
    to_language = st.radio(
        "Select the 'target' language: \n\n",
        ["English", "German", "French", "Italian"],
        key="to_language", index=1, on_change=update_lang,
        horizontal=True, 
    )
    audio_language = st.radio(
        "Select the 'audio' language: \n\n",
        ["English", "German", "French", "Italian"],
        key="audio_language", index=1, 
        horizontal=True,
    )
    st.markdown(f'Learn to speak :orange[{to_language}]')
    
    def update_slider(ikey, skey):
        st.session_state[skey] = st.session_state[ikey]
    def update_num(ikey, skey):
        st.session_state[ikey] = st.session_state[skey]            

    # val = st.number_input('Input', value = 0, key = 'numeric', on_change = update_slider)

    # slider_value = st.slider('slider', min_value = 0, 
    #                         value = val, 
    #                         max_value = 5,
    #                         step = 1,
    #                         key = 'slider', on_change= update_numin)

    def custom_slider_with_input(label, min_value, max_value, value, step, key, help):

        input_value = st.sidebar.number_input(label=label, min_value=min_value, max_value=max_value, value=value, step=step, key=f'i{key}', on_change=update_slider, args=[f'i{key}', f's{key}'], help=help)   

        slider_value = st.sidebar.slider(label=label, label_visibility='hidden', min_value=min_value, max_value=max_value, value=value, step=step, key=f's{key}', on_change=update_num, args=[f'i{key}', f's{key}'])
  
        return input_value

    temp_tt= """ 
        Temperature controls the randomness in token selection

        A lower temperature is good when you expect a true or correct response. A temperature of 0 means the highest probability token is always selected.
        A higher temperature can lead to diverse or unexpected results. Some models have a higher temperature max to encourage more random responses.
        """

    token_tt = """Output token limit determines the maximum amount of text output from one prompt. A token is approximately four characters."""
    
    # temperature = st.sidebar.slider('Temperature', min_value=0.01, max_value=5.0, value=1.0, step=0.01, key='temp')
    temperature = custom_slider_with_input("Temperature", min_value=0.01, max_value=2.0, value=1.0, step=0.01, key='temp', help=temp_tt)
    # top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.95, step=0.01)
    # max_length = st.sidebar.slider('Output token limit', min_value=2048, max_value=8192, value=2048, step=8, key='token')
    max_length = custom_slider_with_input('Output token limit', min_value=2048, max_value=8192, value=2048, step=8, key='token', help=token_tt)

    def clear_history():
        st.session_state.response = ''

    st.sidebar.button('Clear Response', on_click=clear_history)

# temperature = 1.0
top_p = 0.95
# max_length = 2048

st.markdown("*:green[Powered by Vertex AI Gemini]*")
st.subheader("Translate & Perfect Any Text")

# st.markdown(f'Learn to speak :orange[{to_language}]')
context = st.text_area(
    "Context / Scenario: \n\n", key="context", max_chars=2000, placeholder="Type a scenario"
)

task = st.text_area(
    "Content: \n\n", key="task", max_chars=2000, placeholder="Type a sentence"
)
task_response = ''
if task:
    config = {
    "temperature": 0.8,
    "max_output_tokens": 2048,
    }
    prompt_lang = f"""Mention the language in which the text within quotes "{task}" is written in, in just one word"""
    task_response = get_gemini_pro_vision_response( # changed from text_response
                text_model_pro,
                prompt_lang,
                generation_config=config,
            )
    # if task_response:
    #     st.write("The text is in:")
    #     st.write(task_response)


if task_response!= '' and from_language != task_response.strip():
    from_language = task_response

if context.strip() == '':
    context = "general or any"

trans_chk = st.checkbox('Translate Only')
# print('trans_chk', trans_chk)
if trans_chk:
    prompt1 = f"""You need to assist with translating the text within quotes "{task}" to {to_language}, otherwise, if the text provided is in {to_language} then translate the text to {orig_lang}"""
else:
    prompt1 = f"""I want to learn to form grammatically correct sentences in {to_language} based on a {context} scenario / context. You need to assist with translating the text within quotes "{task}" to {to_language}, otherwise, if the text provided is in {to_language}, verify if the text is grammatically correct in {to_language} and and if not grammatically correct, provide the corrected text in {to_language} and explain the reason why the original sentences are not correct along with the correct grammar. Additionally provide alternative sentence constructs with similar meaning."""
config = {
    "temperature": temperature, # 0.8,
    "max_output_tokens": 2048,
}
# print("prompt1:", prompt1)

generate_t2t = st.button("Generate", key="generate_t2t")
if generate_t2t and prompt1:
    # st.write(prompt1)
    with st.spinner("Processing ..."):
        # first_tab1, first_tab2 = st.tabs(["Translation", "Prompt"])
        # with first_tab1:
        response = get_gemini_pro_vision_response( # changed from text_response
            multimodal_model_pro,
            prompt1,
            generation_config=config,
        )
        if response:
            # st.markdown(':red[**Response**]')
            st.divider()
            # st.write(response)

            if "response" not in st.session_state.keys():
                st.session_state.response = response

            st.write(response)

        # generate_audio = st.button("Generate Audio", key="generate_audio", on_click=text_to_wav, args=["de-DE-Neural2-B", response])
        # st.markdown(response)
        match audio_language:
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

        # text_to_wav(speech_lang, response.replace('*', ''))
            # if os.path.isfile('./de-DE-Neural2-B.wav'):
            #     st.audio("de-DE-Neural2-B.wav", format="audio/wav", loop=False)

            # if generate_audio:
            #     print('generating audio')
            #     text_to_wav("de-DE-Neural2-B", response)

        # with first_tab2:
        #     st.markdown(prompt1)




