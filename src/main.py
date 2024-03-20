import sys
sys.path.append('/Users/mrinoyb2/git/FinanceGPT/src/rag_llm.py')
from rag_llm import RAGModel
from evaluate import Evaluate
import os
from dotenv import load_dotenv
import time

# Load the environment variables
load_dotenv()

MONGODB_CONNECTION_STRING = os.getenv("MONGODB_URI")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION")
API_TOKEN = os.getenv("REPLICATE_API_TOKEN")


def chat_mode():
    """
    Funtion to activate chat mode
    
    :return: None
    """
    # Initialize the RAGModel with your MongoDB and Replicate settings
    rag_model = RAGModel(
        mongo_connection_string=MONGODB_CONNECTION_STRING,
        mongo_database=MONGODB_DATABASE,
        mongo_collection=MONGODB_COLLECTION,
        api_token=API_TOKEN
    )

    # Ask the user for their details
    age = int(input("Enter your age: "))
    location = input("Enter your location: ")
    Annual_Income = int(input("Enter your annual income: "))
    Employment_Status = input("Enter your employment status (Employed, Unemployed, Self-employed): ")
    Debt = int(input("Enter the value of your debt: "))
    Assets = int(input("Enter the value of your assets: "))
    Credit_score = int(input("Enter your credit score: "))
    Financial_Goals = input("Enter your financial goals: ")
    Risk_Tolerance = input("Enter your risk tolerance (Low, Medium, High): ")
    Time_Horizon = input("Enter your time horizon (Short-term, Medium-term, Long-term): ")

    # Pass the user's details to the model
    user_input = {
    "Age": age,
    "Location": location,
    "Income": Annual_Income,
    "Employment": Employment_Status,
    "Debt": Debt,
    "Assets": Assets,
    "Credit_Score": Credit_score,
    "Financial_Goals": Financial_Goals,
    "Risk_Tolerance": Risk_Tolerance,
    "Time_Horizon": Time_Horizon
    }   


    # Chat mode
    # Define your query
    query = input("Enter query: ")

    # Perform semantic search and generate an answer
    answer = rag_model.generate_answer(query, user_input)

    # Print the generated answer
    print("Generated Answer:", answer)


def eval_mode():
    """
    Function to activate evaluate mode
    
    :return: None
    """
    # Initialize the RAGModel with your MongoDB and Replicate settings
    rag_model = RAGModel(
        mongo_connection_string=MONGODB_CONNECTION_STRING,
        mongo_database=MONGODB_DATABASE,
        mongo_collection=MONGODB_COLLECTION,
        api_token=API_TOKEN
    )

    # Ask the user for their details
    age = int(input("Enter your age: "))
    location = input("Enter your location: ")
    Annual_Income = int(input("Enter your annual income: "))
    Employment_Status = input("Enter your employment status (Employed, Unemployed, Self-employed): ")
    Debt = int(input("Enter the value of your debt: "))
    Assets = int(input("Enter the value of your assets: "))
    Credit_score = int(input("Enter your credit score: "))
    Financial_Goals = input("Enter your financial goals: ")
    Risk_Tolerance = input("Enter your risk tolerance (Low, Medium, High): ")
    Time_Horizon = input("Enter your time horizon (Short-term, Medium-term, Long-term): ")

    # Pass the user's details to the model
    user_input = {
    "Age": age,
    "Location": location,
    "Income": Annual_Income,
    "Employment": Employment_Status,
    "Debt": Debt,
    "Assets": Assets,
    "Credit_Score": Credit_score,
    "Financial_Goals": Financial_Goals,
    "Risk_Tolerance": Risk_Tolerance,
    "Time_Horizon": Time_Horizon
    }   

    # Initialize the Evaluate class
    eval = Evaluate()

    # Evaluate mode
    # Define your query
    print("You are now in evaluate mode!")

    query = input("Enter query: ")

    # Enter true answer from the text
    true_answer = input("Enter true answer: ")

    # RAG based answer
    rag_answer = rag_model.generate_RAG_answer(query, user_input)

    # Non-RAG based answer
    non_rag_answer = rag_model.generate_non_RAG_answer(query, user_input)

    # Similarity scores
    similarity_scores = eval.calculate_similarity_scores(true_answer, rag_answer, non_rag_answer)

    # Print the generated answer
    print("RAG Answer:", rag_answer)
    print()
    print("Non-RAG Answer:", non_rag_answer)
    print()
    print("Similarity Scores:")
    print()
    print("RAG Answer:", similarity_scores[0])
    print("Non-RAG Answer:", similarity_scores[1])


def main():
    # Welcome message

    # Split the message into three parts
    messages = [
        "Welcome to FinanceGPT💰🚀",
        "Your own conversational AI model that can answer questions related to personal finance.",
        "Type 1 if you want to go to chat mode or 2 if you want to activate evaluate mode."
    ]

    # Loop through each part of the message and print it with a delay
    for msg in messages:
        print(msg)
        time.sleep(1)  # Wait for 1 second before printing the next part

    if input() == "1":
        chat_mode()
    
    else:
        eval_mode()
        

if __name__ == "__main__":
    main()