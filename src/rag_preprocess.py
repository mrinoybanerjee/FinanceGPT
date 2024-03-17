import re
import fitz  # PyMuPDF, used for reading PDF files
import pymongo
from sentence_transformers import SentenceTransformer  # For generating text embeddings
import nltk
from nltk.tokenize import sent_tokenize
import os
from dotenv import load_dotenv

# Ensure the necessary NLTK model is downloaded
nltk.download('punkt', quiet=True)
# Load the environment variables
load_dotenv()

class Preprocess:
    """
    A class for preprocessing PDF documents, storing the processed text in MongoDB,
    and updating the MongoDB documents with sentence embeddings.
    """
    def __init__(self, pdf_directory, text_directory, mongo_connection_string, mongo_database, mongo_collection):
        """
        Initializes the Preprocess object with paths, database settings.
        
        :param pdf_directory: Directory containing PDF documents to be processed.
        :param mongo_connection_string: MongoDB connection string.
        :param mongo_database: Name of the MongoDB database.
        :param mongo_collection: Name of the MongoDB collection.
        """
        self.pdf_directory = pdf_directory
        self.text_directory = text_directory
        self.mongo_connection_string = mongo_connection_string
        self.mongo_database = mongo_database
        self.mongo_collection = mongo_collection
        self.client = pymongo.MongoClient(self.mongo_connection_string)
        self.db = self.client[self.mongo_database]
        self.collection = self.db[self.mongo_collection]
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Load the model for embeddings
    
    def preprocess_mupdf(self, text):
        """
        Preprocesses and cleans the text extracted from a PDF.
        
        :param text: Raw text extracted from the PDF document.
        :return: Cleaned and preprocessed text.
        """
        print("Cleaning text from PDFs...")
        text = re.sub(r'\n\s*\n', '\n', text)  # Remove empty lines
        text = re.sub(r'[^A-Za-z0-9.,;:!?()\'\"\n]+', ' ', text)  # Keep only certain punctuation and alphanumeric characters
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        return text.strip().lower()
    
    def process_pdf_directory(self):
        """
        Processes each PDF file in the directory, extracts and cleans text, 
        then stores the text chunks in MongoDB.
        """
        print("Processing PDFs...")
        for root, dirs, files in os.walk(self.pdf_directory):
            for file in files:
                if file.endswith('.pdf'):
                    self.process_single_pdf(os.path.join(root, file))
    
    def process_single_pdf(self, pdf_path):
        """
        Processes a single PDF file, extracts and cleans text, then stores the text chunks in MongoDB.
        
        :param pdf_path: Path to the PDF document to be processed.
        """
        pdf_document = fitz.open(pdf_path)
        cleaned_text_mupdf = ""
        for page_number in range(pdf_document.page_count):
            page = pdf_document.load_page(page_number)
            text = page.get_text()
            cleaned_text_mupdf += self.preprocess_mupdf(text)
        pdf_document.close()
        
        sentences = self.chunk_by_sentence(cleaned_text_mupdf)
        self.store_chunks_in_mongodb(sentences)

    def process_text_directory(self):
        """
        Processes each text file in the directory, extracts and cleans text, 
        then stores the text chunks in MongoDB.
        """
        print("Processing text files...")
        for root, dirs, files in os.walk(self.text_directory):
            for file in files:
                if file.endswith('.txt'):
                    self.process_single_text(os.path.join(root, file))

    def process_single_text(self, text_path):
        """
        Processes a single text file, extracts and cleans text, then stores the text chunks in MongoDB.
        
        :param text_path: Path to the text file to be processed.
        """
        with open(text_path, 'r') as file:
            text = file.read()
        sentences = self.chunk_by_sentence(text)
        self.store_chunks_in_mongodb(sentences)

    
    def chunk_by_sentence(self, text):
        """
        Splits the text into chunks based on sentence boundaries using NLTK's sent_tokenize.
        
        :param text: Cleaned text to be split into sentences.
        :return: A list of sentences extracted from the text.
        """
        print("Chunking text into sentences...")
        return sent_tokenize(text)
        
    def store_chunks_in_mongodb(self, chunks):
        """
        Stores each chunk of text as a separate document in a MongoDB collection.
        
        :param chunks: List of text chunks (sentences) to be stored.
        """
        print("Storing chunks in MongoDB...")
        for chunk in chunks:
            document = {"text": chunk}
            self.collection.insert_one(document)
        print(f"Total chunks stored in MongoDB: {len(chunks)}")
    
    def update_documents_with_embeddings(self):
        """
        Updates each document in the MongoDB collection with an embedding generated from its text.
        """
        print("Updating documents with sentence embeddings...")
        for document in self.collection.find():
            embedding = self.model.encode(document['text'], convert_to_tensor=False)
            self.collection.update_one({'_id': document['_id']}, {'$set': {'embedding': embedding.tolist()}})
        print("All documents updated with sentence embeddings.")

def main():
    pdf_directory = "/Users/mrinoyb2/git/FinanceGPT/data/pf_books/pdfs"  # Update with the actual path to your PDF directory
    text_directory = "/Users/mrinoyb2/git/FinanceGPT/data/test_folder"  # Update with the actual path to your text directory

    MONGODB_CONNECTION_STRING = os.getenv("MONGODB_URI")
    MONGODB_DATABASE = os.getenv("MONGODB_DATABASE")
    MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION")

    # Initialize the Preprocess object with your parameters
    preprocess_knowledge = Preprocess(pdf_directory=pdf_directory,
                                text_directory=text_directory,
                                   mongo_connection_string=MONGODB_CONNECTION_STRING,
                                   mongo_database=MONGODB_DATABASE,
                                   mongo_collection=MONGODB_COLLECTION)
    
    # Process all texts and PDFs in the specified directory and store their text in MongoDB
    preprocess_knowledge.process_text_directory()
    preprocess_knowledge.process_pdf_directory()
    
    # Update the stored documents with sentence embeddings
    preprocess_knowledge.update_documents_with_embeddings()

if __name__ == "__main__":
    main()