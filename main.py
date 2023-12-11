import streamlit as st
from langchain.llms import OpenAI
from langchain.llms.ollama import Ollama
import json
import langchain
import json
import openai
import os
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, OllamaEmbeddings, HuggingFaceEmbeddings
from langchain.schema import Document

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.agents import AgentExecutor
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.chat_models import ChatOpenAI
import aux
import yaml
import typing
st.title('Team 8️⃣0️⃣')



model_map = yaml.load(open('model_map.yaml'), Loader=yaml.FullLoader)
if 'api_key' not in st.session_state:
    st.session_state['api_key'] = ''
if 'api_retriever' not in st.session_state:
    st.session_state['api_retriever'] = False
if 'example_retriever' not in st.session_state:
    st.session_state['example_retriever'] = False
if 'Ex_Ret' not in st.session_state:
    st.session_state['Ex_Ret'] = None
if 'API_Ret' not in st.session_state:
    st.session_state['API_Ret'] = None
if 'URL' not in st.session_state:
    st.session_state['URL'] = 'http://localhost:8000'



#Util function
def get_embedding_model(modelname : str, model_map : typing.Dict = model_map):
    try:
        model_type = model_map['embedding'][modelname].lower()
        if model_type == 'openai':
            if not st.session_state['api_key'].startswith('sk-'):
                    st.warning(st.session_state['api_key'])
                    st.warning('Please enter your OpenAI API key!', icon='⚠')
            return OpenAIEmbeddings(model=modelname, api_key=st.session_state['api_key'])
        elif model_type == 'ollama':
            return OllamaEmbeddings(model=modelname)
        elif model_type == 'sent':
            return HuggingFaceEmbeddings(model_name=modelname)
        else:
            return KeyError(f"Model Type {model_type} not found")
    except KeyError:
        st.error("Model not found")
        return None

with st.expander("SETUP ⚙️", True):    
    with st.form('Data_Form'):
        url = st.text_input('URL', value=st.session_state['URL'], help='URL of the server where the API is hosted')
        text = st.text_input('API Key', type='password', help='Need it for proprietary model/retriever', value="LOL")
        embedding_model = st.selectbox("Embedding Model", ['text-embedding-ada-002', 'llama2', 'all-MiniLM-L6-v2'])
        api_docs = st.file_uploader("Upload API Docs", type=['json'])
        api_examples = st.file_uploader("Upload API Examples", type=['json'])
        if api_docs:
            api_docs = json.load(api_docs)
        if api_examples:
            api_examples = json.load(api_examples)
        if not api_docs and not api_examples:
            st.warning("Please upload API Docs and Examples if you want to use ToolMode")
            api_docs = []
            api_examples = []

        submitted = st.form_submit_button('Submit')
        if submitted:
            
            st.session_state['URL'] = url
            st.session_state['api_key'] = text
            
            with st.spinner("Loading Embedding Model..."):
                st.session_state['embedding_model'] = get_embedding_model(embedding_model)


            if len(api_docs) > 0 and len(api_examples) > 0:

                with st.spinner("Loading API Descriptions..."):
                    API_descriptions = [
                    Document(page_content=t['description'], metadata={"index": i})
                    for i, t in enumerate(api_docs)
                    ]
                    API_descriptions_vector_store = FAISS.from_documents(
                        API_descriptions, st.session_state['embedding_model'])
                    API_Retriever = API_descriptions_vector_store.as_retriever()
                    st.session_state['api_retriever'] = True
                with st.spinner("Loading Examples.."):
                    API_usage_examples = [
                        Document(page_content=t['Query'], metadata={"index": i})
                    for i, t in enumerate(api_examples)
                    ]
                    API_usage_examples_vector_score = FAISS.from_documents(
                        API_usage_examples, st.session_state['embedding_model'])
                    
                    Examples_Retriever = API_usage_examples_vector_score.as_retriever()
                    st.session_state['example_retriever'] = True
                    st.session_state['Ex_Ret'] = Examples_Retriever
                    st.session_state['API_Ret'] = API_Retriever
            else:
                st.session_state['api_retriever'] = False
                st.session_state['example_retriever'] = False
                st.session_state['Ex_Ret'] = None
                st.session_state['API_Ret'] = None

    if not st.session_state['api_retriever']:
        st.error("No API Descriptions, Can't use ToolMode")
    else:
        st.success("API Description Loaded ✔")

    if not st.session_state['example_retriever']:
        st.error("No Usage Examples, Can't use ToolMode")
    else:
        st.success("Examples Loaded ✔")



with st.container(border=True):
    model_id = st.selectbox("Choose Model", ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'llama2', 'orca-mini', 'wizard-vicuna-uncensored', 'llama2-uncensored'])
        
    use_piro = st.toggle("ToolMode", value=False, help="Toggle this if you want your ToolUse queries answered")


if "messages" not in st.session_state:
    st.session_state.messages = []



if prompt := st.chat_input("What is up?"):
    with st.spinner("Sending Request..."):
        response = aux.post_request(
            st.session_state['URL'],prompt, api_docs, api_examples, st.session_state['api_key'], model_id, use_piro, st.session_state['API_Ret'], st.session_state['Ex_Ret']
        )
        

    if not use_piro:
        with st.chat_message("human"):
            st.session_state.messages.append({"role": "human", "content": prompt})
        with st.chat_message("ai"):
            st.session_state.messages.append({"role": "ai", "content": response['output']['content']})
    else:
        for i, j in response['output'][1]:
            with st.chat_message("human"):
                st.session_state.messages.append({"role": "human", "content": i})
            with st.chat_message("ai"):
                st.session_state.messages.append({"role": "ai", "content": j})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.text(message["content"])

