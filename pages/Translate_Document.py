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

from google.cloud import vision

client_options = {"api_endpoint": "eu-vision.googleapis.com"}
client = vision.ImageAnnotatorClient(client_options=client_options)

PROJECT_ID = 'genai-420915' # os.environ.get("GCP_PROJECT")  # Your Google Cloud Project ID
LOCATION = 'us-central1' # os.environ.get("GCP_REGION")  # Your Google Cloud Project Region
vertexai.init(project=PROJECT_ID, location=LOCATION)

# print(PROJECT_ID, LOCATION)
st.set_page_config(page_title="Translate a document")

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
    st.title('Gemini Bot')
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


st.title("Document Translation")

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

def async_detect_document(gcs_source_uri, gcs_destination_uri):
    """OCR with PDF/TIFF as source files on GCS"""
    import json
    import re
    from google.cloud import vision
    from google.cloud import storage

    PROJECT_ID = 'genai-420915' # os.environ.get("GCP_PROJECT")  # Your Google Cloud Project ID
    LOCATION = 'us-central1' # os.environ.get("GCP_REGION")  # Your Google Cloud Project Region

    # Supported mime_types are: 'application/pdf' and 'image/tiff'
    mime_type = "application/pdf"

    # How many pages should be grouped into each json output file.
    batch_size = 100

    client = vision.ImageAnnotatorClient()

    feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)

    gcs_source = vision.GcsSource(uri=gcs_source_uri)
    input_config = vision.InputConfig(gcs_source=gcs_source, mime_type=mime_type)

    gcs_destination = vision.GcsDestination(uri=gcs_destination_uri)
    output_config = vision.OutputConfig(
        gcs_destination=gcs_destination, batch_size=batch_size
    )

    async_request = vision.AsyncAnnotateFileRequest(
        features=[feature], input_config=input_config, output_config=output_config
    )

    operation = client.async_batch_annotate_files(requests=[async_request])

    print("Waiting for the operation to finish.")
    operation.result(timeout=420)

    # Once the request has completed and the output has been
    # written to GCS, we can list all the output files.
    storage_client = storage.Client(project=PROJECT_ID)

    match = re.match(r"gs://([^/]+)/(.+)", gcs_destination_uri)
    bucket_name = match.group(1)
    prefix = match.group(2)

    bucket = storage_client.get_bucket(bucket_name)

    # List objects with the given prefix, filtering out folders.
    blob_list = [
        blob
        for blob in list(bucket.list_blobs(prefix=prefix))
        if not blob.name.endswith("/")
    ]
    print("Output files:")
    for blob in blob_list:
        print(blob.name)

    # Process the first output file from GCS.
    # Since we specified batch_size=2, the first response contains
    # the first two pages of the input file.
    output = blob_list[0]

    json_string = output.download_as_bytes().decode("utf-8")
    response = json.loads(json_string)

    # The actual response for the first page of the input file.
    first_page_response = response["responses"][0]
    annotation = first_page_response["fullTextAnnotation"]

    # Here we print the full text from the first page.
    # The response contains more information:
    # annotation/pages/blocks/paragraphs/words/symbols
    # including confidence scores and bounding boxes
    print("Full text:\n")
    print(annotation["text"])
    
    return annotation["text"]

# annotation_txt = async_detect_document("gs://vinharith-genai/AIW_short.pdf", "gs://vinharith-genai/AIW/")
read_doc = st.button("Read Document", key="read_doc", on_click=async_detect_document, args=["gs://vinharith-genai/AIW_short.pdf", "gs://vinharith-genai/AIW/"])

