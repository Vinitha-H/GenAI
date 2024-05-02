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

PROJECT_ID = 'genai-420915' # os.environ.get("GCP_PROJECT")  
LOCATION = 'us-central1' # os.environ.get("GCP_REGION")  
vertexai.init(project=PROJECT_ID, location=LOCATION)
st.config._set_option("global.disableWidgetStateDuplicationWarning", True, "test")

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
        try:
            final_response.append(response.text)
        except IndexError:
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
    # print("sound file:", filename)
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
    
    def custom_slider_with_input(label, min_value, max_value, value, step, key, help):

        input_value = st.sidebar.number_input(label=label, min_value=min_value, max_value=max_value, value=value, step=step, key=f'i{key}', on_change=update_slider, args=[f'i{key}', f's{key}'], help=help)   

        slider_value = st.sidebar.slider(label=label, label_visibility='collapsed', min_value=min_value, max_value=max_value, value=value, step=step, key=f's{key}', on_change=update_num, args=[f'i{key}', f's{key}'])
  
        return input_value

    temp_tt= """ 
        Temperature controls the randomness in token selection

        A lower temperature is good when you expect a true or correct response. A temperature of 0 means the highest probability token is always selected.
        A higher temperature can lead to diverse or unexpected results. Some models have a higher temperature max to encourage more random responses.
        """

    token_tt = """Output token limit determines the maximum amount of text output from one prompt. A token is approximately four characters."""
    
    temperature = custom_slider_with_input("Temperature", min_value=0.01, max_value=2.0, value=1.0, step=0.01, key='temp', help=temp_tt)
    max_length = custom_slider_with_input('Output token limit', min_value=2048, max_value=8192, value=2048, step=8, key='token', help=token_tt)

    def clear_history():
        st.session_state.response = ''

    st.sidebar.button('Clear Response', on_click=clear_history)    

# temperature = 1.0
top_p = 0.95
# max_length = 2048

st.markdown("*:green[Powered by Vertex AI Gemini]*")
st.subheader("Translate & Perfect Any Text")

tab1, tab2 = st.tabs(
    ["Text input", "Audio input"]
)

with tab1:
    context = st.text_area(
        "Context / Scenario: \n\n", key="context", max_chars=2000, placeholder="Type a scenario"
    )

    task = st.text_area(
        "Content: \n\n", key="task", max_chars=2000, placeholder="Type a sentence"
    )

    context = context.replace('"', "'")
    task = task.replace('"', "'")

    task_response = ''
    if task:
        config = {
        "temperature": 0.8,
        "max_output_tokens": 2048,
        }
        prompt_task_lang = f"""Mention the language in which the text within quotes "{task}" is written in, in just one word"""
        task_response = get_gemini_pro_vision_response( 
                    text_model_pro,
                    prompt_task_lang,
                    generation_config=config,
                )

    if task_response!= '' and from_language != task_response.strip():
        from_language = task_response

    if context.strip() == '':
        context = "general or any"

    trans_chk = st.checkbox('Content - Translate Only', key='trans_chk')

    prompt1 = f"""I want to learn to form grammatically correct sentences in {to_language} based on a {context} scenario / context. You need to assist with translating the text within quotes "{task}" to {to_language}, otherwise, if the text provided is in {to_language}, verify if the text is grammatically correct in {to_language} and and if not grammatically correct, provide the corrected text in {to_language} and explain the reason why the original sentences are not correct along with the correct grammar. Additionally provide alternative sentence constructs with similar meaning."""

    if trans_chk:
        prompt1 = f"""Translate the text within quotes "{task}" to {to_language}, otherwise, if the text provided is in {to_language} then translate the text to {from_language}"""
        # multimodal_model_pro = 'gemini-1.0-pro-vision'
        use_model = text_model_pro
        top_p = 0.4
    else:
        use_model = multimodal_model_pro
        top_p = 0.95

    config = {
        "temperature": temperature, # 0.8,
        "max_output_tokens": max_length # 2048,
    }

    generate_t2t = st.button("Translate and Explain", key="generate_t2t")
    if generate_t2t and prompt1 != '':
        with st.spinner("Processing ..."):
            response = get_gemini_pro_vision_response( 
                # multimodal_model_pro,
                use_model,
                prompt1,
                generation_config=config,
            )
            if response:
                st.divider()

                if "response" not in st.session_state.keys():
                    st.session_state.response = response

                st.write(response)

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

            text_to_wav(speech_lang, response.replace('*', ''))

with tab2:
    # speech to text
    st.markdown("**Audio transcription and Translation**")
    uploaded_file = st.file_uploader("Choose an audio file", type=['wav'])

    from google.cloud import speech       
    from google.cloud import storage

    def upload_blob(bucket_name, source_file_name, destination_blob_name):
        """Uploads a file to the bucket."""
        # The ID of your GCS bucket
        # bucket_name = "your-bucket-name"
        # The path to your file to upload
        # source_file_name = "local/path/to/file"
        # The ID of your GCS object
        # destination_blob_name = "storage-object-name"

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        # Optional: set a generation-match precondition to avoid potential race conditions
        # and data corruptions. The request to upload is aborted if the object's
        # generation number does not match your precondition. For a destination
        # object that does not yet exist, set the if_generation_match precondition to 0.
        # If the destination object already exists in your bucket, set instead a
        # generation-match precondition using its generation number.
        generation_match_precondition = 1

        # blob.upload_from_filename(source_file_name, if_generation_match=generation_match_precondition)
        blob.upload_from_filename(source_file_name)

        print(
            f"File {source_file_name} uploaded to {destination_blob_name}."
        )

    def transcribe_gcs(gcs_uri: str) -> str:
        """Asynchronously transcribes the audio file specified by the gcs_uri.

        Args:
            gcs_uri: The Google Cloud Storage path to an audio file.

        Returns:
            The generated transcript from the audio file provided.
        """
        client = speech.SpeechClient()

        audio = speech.RecognitionAudio(uri=gcs_uri)
        match audio_language:
            case "English":
                audio_lang = "en-GB"
            case "German":
                audio_lang = "de-DE"
            case "French":
                audio_lang = "fr-FR"
            case "Italian":
                audio_lang = "it-IT"
            case _:
                audio_lang = "en-GB"

        config = speech.RecognitionConfig(
            language_code=audio_lang,
        )

        operation = client.long_running_recognize(config=config, audio=audio)

        # print("Waiting for operation to complete...")
        response = operation.result(timeout=180)

        transcript_builder = []
        # Each result is for a consecutive portion of the audio. Iterate through
        # them to get the transcripts for the entire audio file.
        for result in response.results:
            # The first alternative is the most likely one for this portion.
            transcript_builder.append(f"\n{result.alternatives[0].transcript}")
            # transcript_builder.append(f"\nConfidence: {result.alternatives[0].confidence}")

        transcript = "".join(transcript_builder)

        return transcript

    generate_s2t = st.button("Translate Audio", key="generate_s2t")

    if generate_s2t and uploaded_file is not None:
        import tempfile
        bucket_name = 'vinharith-genai'

        with st.spinner("Processing audio..."):
            temp_dir = tempfile.mkdtemp()
            filepath = os.path.join(temp_dir, uploaded_file.name)
            with open(filepath, "wb") as f:
                    f.write(uploaded_file.getvalue())

            upload_blob(bucket_name, filepath, uploaded_file.name)
        
            audio_transcript = transcribe_gcs(f'gs://{bucket_name}/{uploaded_file.name}')    
            st.write(":orange[Audio Transcript]")
            st.write(audio_transcript)

            # translate
            st.divider()
            st.write(":orange[Audio Translation]")
            audio_prompt = f"""Translate the text {audio_transcript} within quotes to {to_language} otherwise, if the text provided is in {to_language} then translate the text to {from_language}"""
            config = {
                "temperature": temperature, # 0.8,
                "max_output_tokens": max_length, # 2048,
                "top_p": 0.95
            }
            audio_response = get_gemini_pro_vision_response(
                    multimodal_model_pro,
                    audio_prompt,
                    generation_config=config,
                )
            if audio_response:
                st.write(audio_response)
                st.divider()