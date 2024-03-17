import streamlit as st
import os
from dotenv import load_dotenv
from rag_llm import RAGModel

# Load the environment variables
load_dotenv()

MONGODB_CONNECTION_STRING = str(os.getenv("MONGODB_URI"))
MONGODB_DATABASE = str(os.getenv("MONGODB_DATABASE"))
MONGODB_COLLECTION = str(os.getenv("MONGODB_COLLECTION"))
API_TOKEN = str(os.getenv("REPLICATE_API_TOKEN"))

# Initialize the RAGModel with your MongoDB and Replicate settings
# Make sure streamlit is caching the model
@st.cache(allow_output_mutation=True)
def load_model():
    return RAGModel(
        mongo_connection_string=MONGODB_CONNECTION_STRING,
        mongo_database=MONGODB_DATABASE,
        mongo_collection=MONGODB_COLLECTION,
        api_token=API_TOKEN
    )

# Streamlit app customization
st.set_page_config(page_title="FinanceGPT", page_icon="💰", layout="centered", initial_sidebar_state="collapsed")
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       .reportview-container {
            background: #000000
       }
       /* Center the logo */
       .logo-img {
           display: flex;
           justify-content: center;
       }
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

# Add an enter button to submit the query
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #00b8a9;
        color: white;
        font-size: 18px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add a custom footer
footer = """
    <style>
    .reportview-container .main footer {visibility: hidden;}    
    </style>
    """
st.markdown(footer, unsafe_allow_html=True)


# Display logo at the center
st.markdown("<div class='logo-img'>", unsafe_allow_html=True)
st.image("src/.streamlit/AyurGPT.jpeg", width=200)  # Adjust the path and width as needed
st.markdown("</div>", unsafe_allow_html=True)

# Display title and description
st.title("FinanceGPT")
st.write("FinanceGPT is a conversational AI model that can answer questions related to personal finance.")

# Enable users to input personal finance details
st.header("Please answer the following questions to get started:")
age = st.number_input("Age:")
Annual_Income = st.number_input("Annual Income:")
Employment_Status = st.selectbox("Employment Status:", ["Employed", "Unemployed", "Self-Employed"])
Debt = st.number_input("Value of Debt:")
Assets = st.number_input("Value of Assets:")
Credit_score = st.number_input("Credit Score:")
Financial_Goals = st.text_input("What are your financial goals?")
Risk_Tolerance = st.selectbox("What is your risk tolerance?", ["Low", "Medium", "High"])
Time_Horizon = st.selectbox("What is your time horizon?", ["Short-term", "Medium-term", "Long-term"])

# Pass the user's details to the model
user_input = {
    "Age": age,
    "Income": Annual_Income,
    "Employment": Employment_Status,
    "Debt": Debt,
    "Assets": Assets,
    "Credit_Score": Credit_score,
    "Financial_Goals": Financial_Goals,
    "Risk_Tolerance": Risk_Tolerance,
    "Time_Horizon": Time_Horizon
}

# Define your query
query = st.text_input("Message FinanceGPT: ")

# Perform semantic search and generate an answer
rag_model = load_model()
# give users feedback that the model is loading
if query:
    st.write("FinanceGPT is brewing financial wisdom for you...💰🚀")

answer = rag_model.generate_answer(query, user_input)

# Print the generated answer

st.write("Generated Answer:", answer)

