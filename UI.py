import streamlit as st

class UI:

    def __init__(self):
        with st.sidebar:
            self.title = st.title("Prisma")
            self.session_options = ["default"]
            self.model_options = st.selectbox(
                "Select a LLM Model",
                ("LLaMA (Default)", "Gemma", "DeepSeek R1"),
            )
            self.behavior_options = st.selectbox(
                "Select Model Behavior",
                ("Technical (Default)", "Casual"),
            )
            self.uploaded_files = st.file_uploader(label="Upload Your Files", accept_multiple_files=True, type=['pdf', 'csv', 'text', 'txt'])
            st.divider()
            st.info('Custom API Keys are not stored', icon="ℹ️")
            with st.popover("Custom API Keys"):
                st.markdown("Enter Your Custom API Keys 👇")
                self.custom_groq_key = st.text_input("Your Custom Gorq API Key", type='password')
                self.custom_hf_key = st.text_input("Your Custom Hugging Face API Key", type='password')
            st.divider()
            st.markdown("Sessions")
            with st.popover("Manage Sessions"):
                self.current_session = st.text_input(label="Current Session 🕒", placeholder='default', value="default")
                if self.current_session not in self.session_options:
                    self.session_options.append(self.current_session)
                self.session_options = st.selectbox(
                    "Select Session",
                    options=self.session_options,
                )

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        self.prompt = st.chat_input("What is up?")

        if self.prompt:
            # Display user message in chat message container
            st.chat_message("user").markdown(self.prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": self.prompt})
    
    def print_llm_response(self, response):
        if self.prompt:
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
