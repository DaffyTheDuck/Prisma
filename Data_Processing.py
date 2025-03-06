import streamlit as st
import pandas as pd
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from NLP_Processing import NLP_Preprocessing

class Data_Preprocessing:
    def __init__(self, user_files):
        self.user_files = user_files
        self.chunk_size = 1000
        self.chunk_overlap = 100
        self.all_documents = []

    def process_text(self, file):
        """Processes a text file and applies NLP preprocessing."""
        try:
            text_data = file.read().decode("utf-8").strip()  # Read and decode
            if not text_data:
                st.error("Error: The text file is empty.")
                return None
            pre_processing = NLP_Preprocessing(text_data)
            return pre_processing.preprocess_text()
        except UnicodeDecodeError:
            st.error("Error: Unable to decode the text file. Please use UTF-8 encoding.")
            return None
        except Exception as e:
            st.error(f"Error processing text file: {e}")
            return None

    def process_files(self):
        """Processes uploaded files based on their format."""
        if not self.user_files:
            return []

        for user_file in self.user_files:
            st.toast(f"Processing {user_file.name}...")
            file_extension = user_file.name.split(".")[-1].lower()

            # Handling text files
            if file_extension in ["txt", "text"]:
                text_data = self.process_text(user_file)
                if text_data:
                    with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8", suffix=".txt") as temp_file:
                        temp_file.write(text_data)
                        temp_path = temp_file.name  # Get temporary file path

                    loader = TextLoader(temp_path)  # Use temp file path
                    text_docs = loader.load()
                    text_file_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
                    self.all_documents.extend(text_file_splitter.split_documents(text_docs))

            # Handling CSV files
            elif file_extension == "csv":
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
                        temp_file.write(user_file.read())  # Save file content
                        temp_path = temp_file.name  # Get temp file path

                    loader = CSVLoader(temp_path)  # Use temp file path
                    csv_docs = loader.load()
                    csv_file_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
                    self.all_documents.extend(csv_file_splitter.split_documents(csv_docs))

                except Exception as e:
                    st.error(f"Error processing CSV file {user_file.name}: {e}")

            # Handling PDF files
            elif file_extension == "pdf":
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(user_file.read())  # Save file content
                        temp_path = temp_file.name  # Get temp file path

                    loader = PyPDFLoader(temp_path)  # Use temp file path
                    pdf_docs = loader.load()
                    pdf_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
                    self.all_documents.extend(pdf_splitter.split_documents(pdf_docs))

                except Exception as e:
                    st.error(f"Error processing PDF file {user_file.name}: {e}")

        return self.all_documents
