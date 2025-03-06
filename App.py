import os
from dotenv import load_dotenv
import streamlit as st

# File import
from UI import UI
from LLM_Models import LLM_Models
from Data_Processing import Data_Preprocessing

class App:

    def __init__(self):

        # load the env file
        load_dotenv()

        # Load the HF and GroqKey
        self.GROQ_API_KEY = os.environ['GORQ_API_KEY']
        self.HF_TOKEN = os.environ['HF_TOKEN']
        
        self.ui = UI()

        # get custom api keys
        self.user_groq_key = self.ui.custom_groq_key
        self.user_hf_key = self.ui.custom_hf_key

        # get the model behavior
        self.model_persona = self.ui.behavior_options

        # get the user input from the ui
        self.user_input = self.ui.prompt

        self.llm_models = LLM_Models(
            self.user_input, 
            self.GROQ_API_KEY,
            self.HF_TOKEN, 
            self.user_groq_key, 
            self.user_hf_key,
            self.model_persona,
            Data_Preprocessing(self.ui.uploaded_files).process_files()
        )

        # get users choice for LLM model
        self.users_model_choice = self.ui.model_options

        if self.users_model_choice == "Gemma":
            self.ui.print_llm_response(self.llm_models.get_response_from_gemma())
        elif self.users_model_choice == "DeepSeek R1":
            self.ui.print_llm_response(self.llm_models.get_response_from_deepseek())
        else:
            self.ui.print_llm_response(self.llm_models.get_response_from_llama())
            

if __name__ == "__main__":
    App()
