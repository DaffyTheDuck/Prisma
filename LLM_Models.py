from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import streamlit as st
from langchain_community.vectorstores import FAISS

class LLM_Models:
    
    def __init__(self, user_input, GROQ_API, HF_TOKEN, user_groq_key, user_hf_key, model_persona, chunked_docs):
        self.user_input = user_input
        self.GROQ_API = GROQ_API
        self.HF_TOKEN = HF_TOKEN
        self.user_groq_key = user_groq_key
        self.user_hf_key = user_hf_key
        self.model_persona = model_persona
        self.chunked_docs = chunked_docs

        # Personas
        with open("Personas/Casual.txt", "r") as f:
            self.casual_persona = f.read()
            f.close()

        with open("Personas/Technical.txt", "r") as f:
            self.technical_persona = f.read()
            f.close()
        
        # HF Embeddings
        self.embeddings = HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L6-v2")
    
    def get_llm(self, model):
        if not self.user_groq_key:
            self.llm = ChatGroq(
                model=model,
                api_key=self.GROQ_API
            )
            return self.llm
        self.llm = ChatGroq(
            model=model,
            api_key=self.user_groq_key
        )
        return self.llm
    
    def create_chat_template(self):
        if self.model_persona == "Casual":
            self.casual_prompt = ChatPromptTemplate.from_template(f"{self.casual_persona}")
            return self.casual_prompt
        else:
            self.technical_prompt = ChatPromptTemplate.from_template(f"{self.technical_persona}")
            return self.technical_prompt

    def get_response_from_llama(self):
        if self.user_input:
            self.llama_llm = self.get_llm(model="llama-3.3-70b-versatile")
            self.chat_template = self.create_chat_template()
            self.chain =  create_stuff_documents_chain(self.llama_llm, self.chat_template)
            if len(self.chunked_docs) != 0:
                if "vectors" not in st.session_state:
                    st.session_state.embeddings = self.embeddings
                    st.session_state.vectors = FAISS.from_documents(self.chunked_docs, st.session_state.embeddings)
                    self.retriver = st.session_state.vectors.as_retriever()
                    self.retrieval_chain = create_retrieval_chain(self.retriver, self.chain)
                    self.llama_response = self.retrieval_chain.invoke({"input": self.user_input})
                    return self.llama_response['answer']
                self.retriver = st.session_state.vectors.as_retriever()
                self.retrieval_chain = create_retrieval_chain(self.retriver, self.chain)
                self.llama_response = self.retrieval_chain.invoke({"input": self.user_input})
                return self.llama_response['answer']
            else:
                self.llama_response = self.chain.invoke({"input": self.user_input, "context": ""})
                return self.llama_response

    def get_response_from_gemma(self):
        if self.user_input:
            self.gemma_llm = self.get_llm(model="gemma2-9b-it")
            self.chat_template = self.create_chat_template()
            self.chain =  create_stuff_documents_chain(self.gemma_llm, self.chat_template)
            if len(self.chunked_docs) != 0:
                if "vectors" not in st.session_state:
                    st.session_state.embeddings = self.embeddings
                    st.session_state.vectors = FAISS.from_documents(self.chunked_docs, st.session_state.embeddings)
                    self.retriver = st.session_state.vectors.as_retriever()
                    self.retrieval_chain = create_retrieval_chain(self.retriver, self.chain)
                    self.gemma_response = self.retrieval_chain.invoke({"input": self.user_input})
                    return self.gemma_response['answer']
                self.retriver = st.session_state.vectors.as_retriever()
                self.retrieval_chain = create_retrieval_chain(self.retriver, self.chain)
                self.gemma_response = self.retrieval_chain.invoke({"input": self.user_input})
                return self.gemma_response['answer']
            else:
                self.gemma_response = self.chain.invoke({"input": self.user_input, "context": ""})
                return self.gemma_response

    def get_response_from_deepseek(self):
        if self.user_input:
            self.deepseek_llm = self.get_llm(model="deepseek-r1-distill-llama-70b")
            self.chat_template = self.create_chat_template()
            self.chain =  create_stuff_documents_chain(self.deepseek_llm, self.chat_template)
            if len(self.chunked_docs) != 0:
                if "vectors" not in st.session_state:
                    st.session_state.embeddings = self.embeddings
                    st.session_state.vectors = FAISS.from_documents(self.chunked_docs, st.session_state.embeddings)
                    self.retriver = st.session_state.vectors.as_retriever()
                    self.retrieval_chain = create_retrieval_chain(self.retriver, self.chain)
                    self.deepseek_response = self.retrieval_chain.invoke({"input": self.user_input})
                    return self.deepseek_response['answer']
                self.retriver = st.session_state.vectors.as_retriever()
                self.retrieval_chain = create_retrieval_chain(self.retriver, self.chain)
                self.deepseek_response = self.retrieval_chain.invoke({"input": self.user_input})
                return self.deepseek_response['answer']
            else:
                self.deepseek_response = self.chain.invoke({"input": self.user_input, "context": ""})
                return self.deepseek_response
