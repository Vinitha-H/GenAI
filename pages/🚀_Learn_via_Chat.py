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

PROJECT_ID = 'genai-420915' # os.environ.get("GCP_PROJECT")  # Your Google Cloud Project ID
LOCATION = 'us-central1' # os.environ.get("GCP_REGION")  # Your Google Cloud Project Region
vertexai.init(project=PROJECT_ID, location=LOCATION)
st.config._set_option("global.disableWidgetStateDuplicationWarning", True, "test")

# print(PROJECT_ID, LOCATION)
st.set_page_config(page_title="LinguaCoach", page_icon="🚀")

def get_gemini_pro_vision_response(
    model, prompt_list, generation_config={}, stream: bool = True
):
    generation_config = {"temperature": 0.1, "max_output_tokens": 2048}
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

st.header("LinguaCoach - Translate & Correct via Chat", divider="rainbow")
text_model_pro = GenerativeModel("gemini-1.0-pro-vision")

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": help_response}]

from_language = 'English'
to_language = 'German'

def get_help_response():
    help_txt = f"""Translate the text within quotes "How may I assist you today?" into {st.session_state.to_language}"""
    # print("target lang", st.session_state.to_language)

    config = {
    "temperature": 0.8,
    "max_output_tokens": 2048,
    }

    help_response = get_gemini_pro_vision_response(
                text_model_pro,
                help_txt,
                generation_config=config,
            )
    # print("help:", help_response)
    st.session_state.messages = [{"role": "assistant", "content": help_response}]

with st.sidebar:
    st.title('LinguaCoach Chatbot')
    st.subheader('Parameters')
    selected_model = "gemini-1.5-pro-preview-0409"

    # Premise
    from_language = st.radio(
        "Select the 'source' language: \n\n",
        ["English", "German", "French", "Italian"],
        key="from_language",
        horizontal=True,
    )

    to_language = st.radio(
        "Select the 'target' language: \n\n",
        ["English", "German", "French", "Italian"],
        key="to_language", index=1, 
        horizontal=True, on_change=get_help_response
    )
    st.markdown(f'Learn to speak :orange[{to_language}]')

    def update_slider(ikey, skey):
        st.session_state[skey] = st.session_state[ikey]
    def update_num(ikey, skey):
        st.session_state[ikey] = st.session_state[skey]     
    
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
    
    temperature = custom_slider_with_input("Temperature", min_value=0.01, max_value=2.0, value=1.0, step=0.01, key='temp', help=temp_tt)
    max_length = custom_slider_with_input('Output token limit', min_value=2048, max_value=8192, value=2048, step=8, key='token', help=token_tt)

top_p = 0.95

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }
generation_config = {
    "max_output_tokens": max_length, # 8192,
    "temperature": temperature, # 1,
    "top_p": top_p, # 0.95,
    }

# Store LLM generated responses
help_txt = f"""Translate the text within quotes "How may I assist you today?" into {to_language}"""

config = {
"temperature": 0.8,
"max_output_tokens": 2048,
}

help_response = get_gemini_pro_vision_response(
            text_model_pro,
            help_txt,
            generation_config=config,
        )

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": help_response}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

st.sidebar.button('Reset Chat', on_click=clear_chat_history)

def multiturn_generate_content(prompt_input):
    vertexai.init(project="genai-420915", location="us-central1")
    model = GenerativeModel("gemini-1.5-pro-preview-0409",)

    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."

    string_dialogue += f"""I want to learn to form grammatically correct sentences in {st.session_state.to_language} based on the provided text. You need to assist with translating the text within quotes "{prompt_input}" to {st.session_state.to_language}, if the text is not in {st.session_state.to_language}. Otherwise, if the text "{prompt_input}" is in {st.session_state.to_language}, verify if the text is grammatically correct in {st.session_state.to_language} and if it is not grammatically correct, provide the corrected text in {st.session_state.to_language} and explain the reason why the original text is not correct along with the correct grammar."""

    for dict_message in st.session_state.messages:
        # if dict_message["content"]:
        if dict_message["role"] == "user":
            # print("dict_message: ", dict_message["content"])
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"

    # chat = model.start_chat()
    # response = chat.send_message(
    response = model.generate_content(
        f"{string_dialogue} {prompt_input} Assistant: ",
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    answer = response.candidates[0].content.parts[0].text
    return answer

# User-provided prompt        
if (prompt2 := st.chat_input()):
    config = {
    "temperature": 0.8,
    "max_output_tokens": 2048,
    }
    
    prompt_lang = f"""Mention the language in which the text within quotes "{prompt2}" is written in, in just one word"""
    lang_response = get_gemini_pro_vision_response(
                text_model_pro,
                prompt_lang,
                generation_config=config,
            )

    if from_language != lang_response.strip():
        from_language = lang_response

    st.session_state.messages.append({"role": "user", "content": prompt2})
    st.chat_message("user").write(prompt2)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    # with st.chat_message("assistant"):
    with st.spinner("Thinking..."):
        response = multiturn_generate_content(prompt2)
        # placeholder = st.empty()
        # full_response = ''
        # for item in response:
        #     full_response += item
        #     placeholder.markdown(full_response)
        # placeholder.markdown(full_response)
    st.chat_message("assistant").write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
    # time.sleep(0.05)

# st.write("***")

