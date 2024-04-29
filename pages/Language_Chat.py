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

# print(PROJECT_ID, LOCATION)
st.set_page_config(page_title="Learn via chat")

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

st.header("Vertex AI Gemini", divider="rainbow")
text_model_pro = GenerativeModel("gemini-1.0-pro-vision")

with st.sidebar:
    st.title('Gemini Chatbot')
    st.subheader('Models and parameters')
    selected_model = "gemini-1.5-pro-preview-0409"

    # Premise
    from_language = st.radio(
        "Select the language: \n\n",
        ["English", "German"],
        key="from_language",
        horizontal=True,
    )

    to_language = st.radio(
        "Select the language: \n\n",
        ["German", "English"],
        key="to_language",
        horizontal=True,
    )
    st.markdown(f'Learn to speak {to_language}')
    
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=1.0, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.95, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=2048, max_value=8192, value=2048, step=8)


# with tab1:
#     st.write("Using Gemini 1.0 Pro - Text only model")
#     st.subheader("Translate a scenario")

#     st.markdown(f'Learn to speak {to_language}')
    

st.title("Learn a language via Chat")

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
config = {
    "temperature": 0.8,
    "max_output_tokens": 2048,
    }
help_txt = f"""Translate the text within quotes "How may I assist you today?" into {to_language} if {to_language} is different from {from_language} otherwise provide the sentence in the {from_language}"""
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

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": help_response}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

def multiturn_generate_content(prompt_input):
    vertexai.init(project="genai-420915", location="us-central1")
    model = GenerativeModel("gemini-1.5-pro-preview-0409",)

    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."

    string_dialogue += f"""I want to learn to form grammatically correct sentences in {to_language} based on a scenario. You need to assist with translating the sentence within quotes "{prompt_input}" to {to_language}, otherwise, if the sentence is in {to_language}, verify if the sentence is grammatically correct in {to_language} and and if is not grammatically correct, provide the corrected sentence in {to_language} and explain the reason why the original sentence is not correct along with the correct grammar."""

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

