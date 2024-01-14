# import streamlit as st
# from dotenv import load_dotenv
# from langchain.llms import HuggingFaceHub 
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate

# # Load environment variables
# load_dotenv()

# hub_llm = HuggingFaceHub(
#     repo_id='mistralai/Mistral-7B-v0.1',
#     model_kwargs={'temperature': 0.7, 'max_length': 100}
# )

# prompt = PromptTemplate(
#     input_variables=["Zodiac"],
#     template="Your zodiac sign is {Zodiac} which means your future is very bright but you need to be very careful of your surroundings"
# )

# hub_chain = LLMChain(prompt=prompt, llm=hub_llm, verbose=True)
# print(hub_chain.run("Saggitarius"))


import streamlit as st
from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub 
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Function to run the model and display the result
def run_model(zodiac_sign):
    hub_llm = HuggingFaceHub(
        repo_id='mistralai/Mistral-7B-v0.1',
        model_kwargs={'temperature': 0.7, 'max_length': 1000}
    )

    prompt = PromptTemplate(
        input_variables=["Zodiac"],
        template="Your zodiac sign is {Zodiac} which means your future is very bright but you need to be very careful of your surroundings"
    )

    hub_chain = LLMChain(prompt=prompt, llm=hub_llm, verbose=True)
    return hub_chain.run(zodiac_sign)

# Streamlit UI
st.title("Zodiac Sign Future Prediction")

# Get user input
user_zodiac = st.text_input("Enter your zodiac sign")

# Predict the future on button click
if st.button("Predict"):
    if user_zodiac:
        prediction = run_model(user_zodiac)
        st.write(f"Prediction for {user_zodiac}: {prediction}")
    else:
        st.warning("Please enter your zodiac sign.")

