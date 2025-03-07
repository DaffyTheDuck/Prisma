from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import streamlit as st
from langchain_community.vectorstores import FAISS

class LLM_Models:
    
    def __init__(self, user_input, GROQ_API, HF_TOKEN, user_groq_key, user_hf_key, model_persona, chunked_docs, session_options):
        self.user_input = user_input
        self.GROQ_API = GROQ_API
        self.HF_TOKEN = HF_TOKEN
        self.user_groq_key = user_groq_key
        self.user_hf_key = user_hf_key
        self.model_persona = model_persona
        self.chunked_docs = chunked_docs
        self.session_options = session_options

        # Session store
        if 'store' not in st.session_state:
            st.session_state.store = {}

        # Personas
        with open("Prompt_Templates/Personas/Casual.txt", "r") as f:
            self.casual_persona = f.read().strip()
            f.close()

        with open("Prompt_Templates/Personas/Technical.txt", "r") as f:
            self.technical_persona = f.read().strip()
            f.close()
        
        # Chat history template
        with open("Prompt_Templates/ChatHistory.txt", "r") as f:
            self.history_context = f.read().strip()
            f.close()

        # HF Embeddings
        self.embeddings = HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L6-v2")

        # General history context prompt
        self.general_history_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.history_context),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    def get_session_hisotry(self, session_id)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id]=ChatMessageHistory()
        return st.session_state.store[session_id]
    
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
            # self.casual_prompt = ChatPromptTemplate.from_template(f"{self.casual_persona}")
            self.casual_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self.casual_persona),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            return self.casual_prompt
        else:
            # self.technical_prompt = ChatPromptTemplate.from_template(f"{self.technical_persona}")
            self.technical_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self.technical_persona),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            return self.technical_prompt
    
    def create_history_aware_retriever(self, llm, retriever):
        return create_history_aware_retriever(llm=llm, retriever=retriever, prompt=self.general_history_template)

    def get_response_from_llama(self):
        if self.user_input:
            self.llama_llm = self.get_llm(model="llama-3.3-70b-versatile")
            self.chat_template = self.create_chat_template()
            self.chain =  create_stuff_documents_chain(self.llama_llm, self.chat_template)
            if len(self.chunked_docs) != 0:
                if "vectors" not in st.session_state:
                    st.session_state.embeddings = self.embeddings
                    st.session_state.vectors = FAISS.from_documents(self.chunked_docs, st.session_state.embeddings)
                self.retriever = st.session_state.vectors.as_retriever()
                self.history_aware_retriever = self.create_history_aware_retriever(self.llama_llm, self.retriever)
                self.retrieval_chain = create_retrieval_chain(self.history_aware_retriever, self.chain)
                
                self.conv_rag_chain = RunnableWithMessageHistory(
                    self.retrieval_chain, self.get_session_hisotry,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer"
                )
                
                self.llama_response = self.conv_rag_chain.invoke(
                    {"input": self.user_input},
                    config={
                        "configurable": {"session_id": self.session_options}
                    }
                )
                return self.llama_response['answer']
            else:
                self.conv_rag_chain = RunnableWithMessageHistory(
                    runnable=self.chain,
                    get_session_history=self.get_session_hisotry,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer"
                )
                self.llama_response = self.conv_rag_chain.invoke(
                    {"input": self.user_input, "context": ""},
                    config={
                        "configurable": {"session_id": self.session_options}
                    }
                )
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
                self.retriever = st.session_state.vectors.as_retriever()
                self.history_aware_retriever = self.create_history_aware_retriever(self.gemma_llm, self.retriever)
                self.retrieval_chain = create_retrieval_chain(self.history_aware_retriever, self.chain)
                
                self.conv_rag_chain = RunnableWithMessageHistory(
                    self.retrieval_chain, self.get_session_hisotry,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer"
                )
                
                self.gemma_response = self.conv_rag_chain.invoke(
                    {"input": self.user_input},
                    config={
                        "configurable": {"session_id": self.session_options}
                    }
                )
                return self.gemma_response['answer']
            else:
                self.conv_rag_chain = RunnableWithMessageHistory(
                    runnable=self.chain,
                    get_session_history=self.get_session_hisotry,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer"
                )
                self.gemma_response = self.conv_rag_chain.invoke(
                    {"input": self.user_input, "context": ""},
                    config={
                        "configurable": {"session_id": self.session_options}
                    }
                )
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
                self.retriever = st.session_state.vectors.as_retriever()
                self.history_aware_retriever = self.create_history_aware_retriever(self.deepseek_llm, self.retriever)
                self.retrieval_chain = create_retrieval_chain(self.history_aware_retriever, self.chain)
                
                self.conv_rag_chain = RunnableWithMessageHistory(
                    self.retrieval_chain, self.get_session_hisotry,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer"
                )
                
                self.deepseek_response = self.conv_rag_chain.invoke(
                    {"input": self.user_input},
                    config={
                        "configurable": {"session_id": self.session_options}
                    }
                )
                return self.deepseek_response['answer']
            else:
                self.conv_rag_chain = RunnableWithMessageHistory(
                    runnable=self.chain,
                    get_session_history=self.get_session_hisotry,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer"
                )
                self.deepseek_response = self.conv_rag_chain.invoke(
                    {"input": self.user_input, "context": ""},
                    config={
                        "configurable": {"session_id": self.session_options}
                    }
                )
                return self.deepseek_response
