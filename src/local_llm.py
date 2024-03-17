from langchain_community.llms import Ollama
import pandas as pd
import re
from typing import List
from pydantic import BaseModel, validator

# Pydantic model for validating categorized transactions
class CategorizedTransaction(BaseModel):
    date: str
    description: str
    amount: str
    category: str

    @validator('category')
    def category_must_be_short(cls, value):
        if len(value.split()) > 3:
            raise ValueError('Category names must be less than 3 words')
        return value

class FinanceAnalyzer:
    def __init__(self, model_name="finance_gpt_llama2"):
        self.llm = Ollama(model=model_name)
        self.df = None

    def read_transaction_data(self, file_path):
        print(f"Reading transaction data from {file_path}")
        self.df = pd.read_csv(file_path)
        return self.df

    def clean_and_normalize(self, description_column='Description'):
        print(f"Cleaning and normalizing transaction descriptions in column {description_column}")
        if self.df is not None and description_column in self.df.columns:
            self.df[description_column] = self.df[description_column].apply(self._clean_normalize_description)
        else:
            raise ValueError(f"DataFrame is None or column {description_column} not found.")

    def _clean_normalize_description(self, description):
        description = str(description).lower()
        description = re.sub(r'\s+', ' ', description)
        description = re.sub(r'[^a-z0-9\s]', '', description)
        description = re.sub(r'\d', '', description)
        description = ' '.join(description.split()[:3])
        description = re.sub(r'\b(?:al|ak|az|ar|ca|co|ct|de|fl|ga|hi|id|il|in|ia|ks|ky|la|me|md|ma|mi|mn|ms|mo|mt|ne|nv|nh|nj|nm|ny|nc|nd|oh|ok|or|pa|ri|sc|sd|tn|tx|ut|vt|va|wa|wv|wi|wy)\b', '', description)
        description = description.replace('-', ' ').strip()
        return description

    def categorize_transactions(self):
        if self.df is None:
            raise ValueError("DataFrame is None. Ensure transaction data is loaded.")
        
        transactions = self.df.apply(lambda x: f"{x['Date']} - {x['Description']} - {x['Amount']}", axis=1).tolist()
        categorized_df = self._categorize_with_llm(transactions)
        
        self.df = pd.merge(self.df, categorized_df, on=['Date', 'Description', 'Amount'], how='left')

    def _categorize_with_llm(self, transactions: List[str]):
        print("Categorizing transactions using the Ollama...(this may take a while)")
        prompt = "You are a financial advisor. Can you assign an appropriate category to each transaction. Maintain the format: Date - Description - Amount - Category. Category names should be consistent and less than 3 words: \n" + "\n".join(transactions)
        response = self.llm.invoke(prompt)
        
        lines = response.split('\n')
        categorized_transactions = []
        
        for line in lines:
            try:
                date, description, amount, category = line.split(" - ")
                ct = CategorizedTransaction(date=date, description=description, amount=amount, category=category)
                categorized_transactions.append(ct.dict())
            except ValueError:
                print(f"Skipping invalid line: {line}")
            except Exception as e:
                print(f"Validation error: {e}")
        
        return pd.DataFrame(categorized_transactions)

def main():
    # Initialize the finance analyzer with the specified model
    analyzer = FinanceAnalyzer(model_name="finance_gpt_llama2")
    
    # Read transaction data from a CSV file
    transactions_path = "/Users/mrinoyb2/git/FinanceGPT/data/transactions/amex_2023.csv"  # Update this path
    analyzer.read_transaction_data(transactions_path)
    
    # Clean and normalize transaction descriptions
    analyzer.clean_and_normalize(description_column='Description')
    
    # Categorize transactions using the language model
    analyzer.categorize_transactions()
    
    # Export the categorized transactions to a CSV file
    export_path = "/Users/mrinoyb2/git/FinanceGPT/data/llm_analyzed_transactions/categorized_transactions.csv"  # Update this path
    analyzer.df.to_csv(export_path, index=False)

if __name__ == "__main__":
    main()
