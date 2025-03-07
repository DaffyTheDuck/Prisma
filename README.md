# Setup Instructions
1. Clone the repository into your local machine using `git clone git@github.com:DaffyTheDuck/Prisma.git`
2. In a terminal navigate to the cloned repository
3. It is recommended that you create a virtual environment to prevent any requirement conflicts.
4. The virtual env on Linux can be created by `python3 -m venv .venv` and can be activated using `source .venv/bin/activate`
5. Once activated, install the requirements by using `pip install -r requirement.txt`
6. Start the UI by `streamlit run src/App.py`
7. This will open the streamlit UI in the browser e.g. localhost:8501
8. You are good to go now

---
# Architecure of Application
![Architecture of Application]()

---
## NOTE: THE USER NEEDS TO PROVIDE THEIR OWN API KEYS IN AN ORDER FOR THE APPLICATION TO WORK
The API Keys can be generated on [GroqAPI](https://console.groq.com/keys) and [HuggingFace](https://huggingface.co/)
