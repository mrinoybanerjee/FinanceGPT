import pymongo
import numpy as np
from scipy.spatial.distance import cosine
from replicate.client import Client
import google.generativeai as genai  # Import Google GenAI
from sentence_transformers import SentenceTransformer

class RAGModel:
    """
    A class to perform Retrieval-Augmented Generation (RAG) for generating answers
    based on semantic search within a MongoDB database and using the Gemini Pro model
    for text generation.
    """
    def __init__(self, mongo_connection_string, mongo_database, mongo_collection, api_token):
        """
        Initializes the RAGModel with MongoDB connection settings and the API token for Gemini Pro.
        
        :param mongo_connection_string: MongoDB connection string.
        :param mongo_database: MongoDB database name.
        :param mongo_collection: MongoDB collection name.
        :param api_token: Gemini API token.
        """
        self.mongo_connection_string = mongo_connection_string
        self.mongo_database = mongo_database
        self.mongo_collection = mongo_collection
        self.api_token = api_token
        self.client = pymongo.MongoClient(self.mongo_connection_string)
        self.db = self.client[self.mongo_database]
        self.chunks_collection = self.db[self.mongo_collection]
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Load sentence transformer model for embeddings
        genai.configure(api_key=self.api_token)  # Configure GenAI with the provided API key
        self.engineered_context = '''Persona: You are a highly knowledgeable, solution-oriented and personable financial advisor with access to a vast database of financial resources. 
                You prioritize clear, actionable advice tailored to the user's unique situation and goals.
                
                Example 1:
                Question: "I'm 30 years old making $60,000 a year with no debt and $20,000 in savings. How can I start investing for retirement?"
                User Context: Financial Situation: Age: 30, income: 60000, employment: "Full Time", debt: 0, assets: 20000. Financial Goals: Long-term (retirement). Risk Tolerance: Moderate. Life Events: N/A.
                Answer: "Given your stable income, no debt, and a moderate risk tolerance, you're in a great position to start investing for retirement. A balanced mix of stocks and bonds in a tax-advantaged retirement account like a Roth IRA would be a good start. Consider allocating 70% to a diversified stock fund and 30% to bonds. Adjust the allocation as you age or as your risk tolerance changes."

                Example 2:
                Question: "I'm 22, just started my first job earning $45,000, and have $10,000 in student loans. What's my best strategy for saving?"
                User Context: Financial Situation: Age: 22, income: 45000, employment: "Full Time", debt: 10000, assets: 5000. Financial Goals: Short-term (emergency fund), mid-term (debt repayment). Risk Tolerance: Low. Life Events: Starting first job.
                Answer: "Starting with your student loans and building an emergency fund are your first steps. Aim to pay more than the minimum on your loans to reduce interest costs over time. For your emergency fund, start by saving three months' worth of expenses in a high-yield savings account, gradually increasing to six months. Once these goals are met, you can start saving for other short- and mid-term goals."

                Response Style: 
                * Clarity: Explain complex concepts in simple terms.
                * Actionable: Provide specific recommendations and next steps that will solve the problem at hand. 
                * Personalization: Tailor the advice to the user's situation and goals.
                * Transparency: Acknowledge limitations and suggest further resources if needed.

                '''

    def semantic_search(self, query, top_k=5):
        """
        Performs semantic search to find the most relevant text chunks based on the query.
        
        :param query: The query string for which to find relevant documents.
        :param top_k: The number of top results to return.
        :return: A list of tuples containing the document ID, similarity score, and text for each result.
        """
        query_embedding = self.model.encode(query, convert_to_tensor=False)
        similarities = []
        for document in self.chunks_collection.find():
            doc_embedding = np.array(document['embedding'])
            similarity = 1 - cosine(query_embedding, doc_embedding)
            similarities.append((document['_id'], similarity, document['text']))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    

    def generate_answer(self, question, user_input, max_context_length=1000):
        """
        Generates an answer to a given question using the best context found by semantic search
        and a mixtral of experts model.
        
        :param question: The question to generate an answer for.
        :param max_context_length: Maximum length of the context to be considered.
        :return: Generated answer as a string.
        """
        context_results = self.semantic_search(question, top_k=5)
        if context_results:
            # Concatenate the top-k context results into a single string
            context = " ".join([result[2] for result in context_results])
            if len(context) > max_context_length:
                context = context[:max_context_length]
            # Use engineered context with the best context found by semantic search
            prompt = f'''[INST]\n{self.engineered_context}\nContext: {context}\n
                Your Question: "{question}"
                Your User Context: Financial Situation: Age: {user_input['Age']}, income: {user_input['Income']}, employment: "{user_input['Employment']}", 
                debt: {user_input['Debt']}, assets: {user_input['Assets']}, credit score: {user_input['Credit_Score']}, 
                Financial Goals: {user_input['Financial_Goals']}, Risk Tolerance: "{user_input['Risk_Tolerance']}", Time Horizon: "{user_input['Time_Horizon']}".

            [/INST]'''
        else:
            # Use only the engineered context
            prompt = f'''[INST]\n{self.engineered_context}\n
                Your Question: "{question}"
                Your User Context: Financial Situation: Age: {user_input['Age']}, income: {user_input['Income']}, employment: "{user_input['Employment']}", 
                debt: {user_input['Debt']}, assets: {user_input['Assets']}, credit score: {user_input['Credit_Score']}, 
                Financial Goals: {user_input['Financial_Goals']}, Risk Tolerance: "{user_input['Risk_Tolerance']}", Time Horizon: "{user_input['Time_Horizon']}".

            [/INST]'''

        replicate_client = Client(api_token=self.api_token)
         # The mistralai/mixtral-8x7b-instruct-v0.1 model can stream output as it's running.
        for event in replicate_client.stream(
            "mistralai/mixtral-8x7b-instruct-v0.1",
            input={
                "top_k": 50,
                "top_p": 1,
                "prompt": prompt,
                "temperature": 0.5,
                "max_new_tokens": 1024,
                "prompt_template": "<s>[INST] {prompt} [/INST] ",
                "presence_penalty": 0,
                "frequency_penalty": 0
            },
        ):
            print(str(event), end="")
    

    def generate_RAG_answer(self, question, user_input, max_context_length=20000):
        """
        Generates an answer to a given question using the best context found by RAG based semantic search
        and the Gemini Pro model.
        
        :param question: The question to generate an answer for.
        :param max_context_length: Maximum length of the context to be considered.
        :return: Generated answer as a string.
        """
        context_results = self.semantic_search(question, top_k=5)
        # Concatenate the top-k context results into a single string
        context = " ".join([result[2] for result in context_results])
        if len(context) > max_context_length:
            context = context[:max_context_length]
        # Use engineered context with the best context found by semantic search
        prompt = f'''[INST]\n{self.engineered_context}\nContext: {context}\n
                Your Question: "{question}"
                Your User Context: Financial Situation: Age: {user_input['Age']}, income: {user_input['Income']}, employment: "{user_input['Employment']}", 
                debt: {user_input['Debt']}, assets: {user_input['Assets']}, credit score: {user_input['Credit_Score']}, 
                Financial Goals: {user_input['Financial_Goals']}, Risk Tolerance: "{user_input['Risk_Tolerance']}", Time Horizon: "{user_input['Time_Horizon']}".

            [/INST]'''

        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)

        # Handle the case where the response is empty
        if not response.text:
            return "Sorry, I don't have an answer for that."
        
        return response.text
    

    def generate_non_RAG_answer(self, question, user_input):
        """
        Generates an answer to a given question using only the Gemini Pro model.
        
        :param question: The question to generate an answer for.
        :return: Generated answer as a string.
        """
        # Use only the engineered context
        prompt = f'''[INST]\n{self.engineered_context}\n

            Your Question: "{question}"
            Your User Context: Financial Situation: Age: {user_input['Age']}, income: {user_input['Income']}, employment: "{user_input['Employment']}", 
            debt: {user_input['Debt']}, assets: {user_input['Assets']}, credit score: {user_input['Credit_Score']}, 
            Financial Goals: {user_input['Financial_Goals']}, Risk Tolerance: "{user_input['Risk_Tolerance']}", Time Horizon: "{user_input['Time_Horizon']}".

        [/INST]'''


        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)

        # Handle the case where the response is empty
        if not response.text:
            return "Sorry, I don't have an answer for that."
        
        return response.text
