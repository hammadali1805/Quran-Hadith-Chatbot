import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma  # Or your chosen vector store
from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import(‘pysqlite3’)
import sys

sys.modules[‘sqlite3’] = sys.modules.pop(‘pysqlite3’)
st.set_page_config('Quran & Hadith Chatbot')

def load_model(GOOGLE_API_KEY):

    genai.configure(api_key=GOOGLE_API_KEY)

    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3, api_key=GOOGLE_API_KEY)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    loaded_vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)


    query_expansion_template = """You are an AI assistant tasked with improving search queries. Given the original query, please reformulate it in English language to be more specific and comprehensive. The goal is to create a query that will yield the most relevant and accurate results in a similarity search from a vectorstore of English Translation of Quran and Hadiths.

    Original query: {original_query}

    Expanded query:"""

    query_expansion_prompt = PromptTemplate(
        input_variables=["original_query"],
        template=query_expansion_template
    )

    query_expansion_chain = LLMChain(llm=model, prompt=query_expansion_prompt)


    query_result_template = """You are an AI assistant tasked with providing accurate answers based on the given context and your existing knowledge. Avoid providing incorrect or inappropriate answers.

    Context: {context}

    Query: {query}

    Answer:"""

    query_result_prompt = PromptTemplate(
        input_variables=["context", "query"],
        template=query_result_template
    )

    query_result_chain = LLMChain(llm=model, prompt=query_result_prompt)

    st.session_state.model = model
    st.session_state.embeddings = embeddings
    st.session_state.loaded_vectorstore = loaded_vectorstore
    st.session_state.query_result_chain = query_result_chain
    st.session_state.query_expansion_chain = query_expansion_chain


def expand_query(original_query):
    expanded_query = st.session_state.query_expansion_chain.run(original_query=original_query)
    return expanded_query.strip()

def enhanced_similarity_search(original_query, k=3):
    expanded_query = expand_query(original_query)
    print(expanded_query)
    results = st.session_state.loaded_vectorstore.similarity_search(expanded_query, k=k)
    return expanded_query, results

def get_answer(original_query):
    try:
        expanded_query, results = enhanced_similarity_search(original_query, k=10)
        context = str()
        for result in results:
            context+=result.page_content

        print(f'\n\n{context}')
        answer = st.session_state.query_result_chain.run(context=context, query=expanded_query)
        if not answer.strip():
            answer = "Unable to respond to your query at the moment."
    except Exception as e:
        answer = f'Error: {str(e)}\nTry again after sometime or check your API Key.'
    return answer.strip()


st.title("Quran & Hadith ChatBot")

# Check if API key is in session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''

# API Key input
if not st.session_state.api_key:
    st.warning("We don't store any API KEY. Your API KEY will be deleted one you leave the site or reload it.")
    api_key = st.text_input("Enter your Gemini API key:", type="password")
    link = '[Get Free Gemini API KEY](https://aistudio.google.com/app/apikey)'
    st.markdown(link, unsafe_allow_html=True)
    if st.button("Submit API Key"):
        st.session_state.api_key = api_key
        st.rerun()

# Chatbot interface
if st.session_state.api_key:
    if "model" not in st.session_state:
        with st.spinner('Loading Database and ChatBot...'):
            load_model(st.session_state.api_key)

    # Language selection
    # languages = ["English", "Spanish", "French", "German", "Chinese"]
    # selected_language = st.selectbox("Choose a language:", languages)

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Fetching..."):
                response = get_answer(prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
