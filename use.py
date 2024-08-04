from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma  # Or your chosen vector store
from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

GOOGLE_API_KEY = ""
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


query_result_template = """You are an AI assistant tasked with providing accurate answers based on the given context and your existing knowledge. If the answer is not found in either the context or your knowledge base, respond with "Don't know." Avoid providing incorrect or inappropriate answers.

Context: {context}

Query: {query}

Answer:"""

query_result_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template=query_result_template
)

query_result_chain = LLMChain(llm=model, prompt=query_result_prompt)

def expand_query(original_query):
    expanded_query = query_expansion_chain.run(original_query=original_query)
    return expanded_query.strip()

def enhanced_similarity_search(original_query, k=3):
    expanded_query = expand_query(original_query)
    results = loaded_vectorstore.similarity_search(expanded_query, k=k)
    return expanded_query, results

def get_answer(original_query):
    expanded_query, results = enhanced_similarity_search(original_query, k=10)
    print(results)
    context = str()
    for result in results:
        context+=result.page_content

    answer = query_result_chain.run(context=context, query=expanded_query)

    return answer.strip()


