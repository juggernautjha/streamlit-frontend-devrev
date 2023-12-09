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
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.agents import AgentExecutor
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.chat_models import ChatOpenAI
import aux
st.title('Team 8️⃣0️⃣')


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



with st.expander("SETUP ⚙️", True):    
    with st.form('Data_Form'):
        text = st.text_input('API Key', type='password', help='Need it for proprietary model/retriever', value="LOL")
        api_docs = st.file_uploader("Upload API Docs", type=['json'])
        if api_docs:
            api_docs = json.load(api_docs)
        api_examples = st.file_uploader("Upload API Examples", type=['json'])
        if api_examples:
            api_examples = json.load(api_examples)

        submitted = st.form_submit_button('Submit')
        if submitted:
            st.session_state['api_key'] = text
            if not st.session_state['api_key'].startswith('sk-'):
                st.warning(st.session_state['api_key'])
                st.warning('Please enter your OpenAI API key!', icon='⚠')

            with st.spinner("Loading API Descriptions..."):
                API_descriptions = [
                Document(page_content=t['description'], metadata={"index": i})
                for i, t in enumerate(api_docs)
                ]
                API_descriptions_vector_store = FAISS.from_documents(
                    API_descriptions, OpenAIEmbeddings(api_key=st.session_state['api_key']))
                API_Retriever = API_descriptions_vector_store.as_retriever()
                st.session_state['api_retriever'] = True
            with st.spinner("Loading Examples.."):
                API_usage_examples = [
                
                    Document(page_content=t['Query'], metadata={"index": i})
                for i, t in enumerate(api_examples)
                ]
                API_usage_examples_vector_score = FAISS.from_documents(
                    API_usage_examples, OpenAIEmbeddings(api_key=st.session_state['api_key']))
                
                Examples_Retriever = API_usage_examples_vector_score.as_retriever()
                st.session_state['example_retriever'] = True
                st.session_state['Ex_Ret'] = Examples_Retriever
                st.session_state['API_Ret'] = API_Retriever
    if not st.session_state['api_key'].startswith('sk'):
        st.error("Invalid OpenAI Key")
    else:
        st.success("Valid Key Found!!")

    if not st.session_state['api_retriever']:
        st.error("Please upload API Descriptions")
    else:
        st.success("API Description Loaded ✔")

    if not st.session_state['example_retriever']:
        st.error("Please upload Examples")
    else:
        st.success("Examples Loaded ✔")



with st.container(border=True):
    model_id = st.selectbox("Choose Model", ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'llama2'])
        
    use_piro = st.toggle("Get JSON", value=True, help="Toggle this if youo want JSON for your query and don't want to chat :smile:")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})    
    if not (st.session_state['api_key'].startswith('sk-') and st.session_state['api_retriever'] and st.session_state['example_retriever']):
        st.error("Invalid params, please recheck Setup")
    else:
        with st.spinner("Sending Request..."):
            response = aux.post_request(
                prompt, api_docs, api_examples, st.session_state['api_key'], model_id, use_piro, st.session_state['API_Ret'], st.session_state['Ex_Ret']
            )
        

    if not use_piro:
        with st.chat_message("human"):
            st.markdown(prompt)
        with st.chat_message("ai"):
            st.markdown(response['output']['content'])
    else:
        for i, j in response['output'][1]:
            with st.chat_message("human"):
                st.markdown(i)
            with st.chat_message("ai"):
                st.markdown(j)



    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
