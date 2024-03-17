from langchain_community.llms import Ollama
import pandas as pd
import re

class FinanceAnalyzer:
    """
    A class to analyze and process financial transactions with the help of a language model.

    Attributes:
        llm (Ollama): A language model for finance-related inquiries.
        df (DataFrame): DataFrame holding transaction data.
    """

    def __init__(self, model_name="finance_gpt_llama2"):
        """
        Initializes the FinanceAnalyzer with a specified language model.

        Args:
            model_name (str): Name of the language model.
        """
        self.llm = Ollama(model=model_name)
        self.df = None

    def read_transaction_data(self, file_path):
        """
        Reads transaction data from a specified CSV file into a DataFrame.

        Args:
            file_path (str): Path to the CSV file containing transaction data.

        Returns:
            DataFrame: The loaded transaction data.
        """
        self.df = pd.read_csv(file_path)
        print(self.df.head())
        return self.df

    def clean_and_normalize(self, description_column='Description'):
        """
        Cleans and normalizes the transaction descriptions.

        Args:
            description_column (str): The name of the column containing transaction descriptions.
        """
        if self.df is not None and description_column in self.df.columns:
            self.df[description_column] = self.df[description_column].apply(self._clean_normalize_description)
        else:
            raise ValueError(f"DataFrame is None or column {description_column} not found.")

    def _clean_normalize_description(self, description):
        """
        Cleans and normalizes a single transaction description.

        Args:
            description (str): A transaction description.

        Returns:
            str: The cleaned and normalized description.
        """
        description = str(description).lower()
        description = re.sub(r'\s+', ' ', description)
        description = re.sub(r'[^a-z0-9\s]', '', description)
        description = re.sub(r'\d', '', description)
        description = ' '.join(description.split()[:3])
        description = re.sub(r'\b(?:al|ak|az|ar|ca|co|ct|de|fl|ga|hi|id|il|in|ia|ks|ky|la|me|md|ma|mi|mn|ms|mo|mt|ne|nv|nh|nj|nm|ny|nc|nd|oh|ok|or|pa|ri|sc|sd|tn|tx|ut|vt|va|wa|wv|wi|wy)\b', '', description)
        description = description.replace('-', ' ').strip()
        return description

    def categorize_transactions(self):
        """
        Categorizes transactions using the language model. Standardizes category names after assignment.
        """
        if self.df is None:
            raise ValueError("DataFrame is None. Ensure transaction data is loaded.")
        
        # Assuming self.df has columns ['Date', 'Description', 'Amount']
        transaction_descriptions = self.df.apply(lambda row: f"{row['Date']} - {row['Description']} - {row['Amount']}", axis=1)
        transaction_names = ','.join(transaction_descriptions.tolist())
        
        # Call the language model to categorize transactions
        categories_df_all = self._categorize_with_llm(transaction_names)
        
        # Merge categories_df_all back into self.df based on matching criteria
        # This requires the original DataFrame to have a unique way to match transactions back after categorization,
        # for simplicity, it's assumed that the transactions can be matched based on existing columns.
        self.df = pd.merge(self.df, categories_df_all, on=['Date', 'Description', 'Amount'], how='left')
        print(self.df.head())

    def _categorize_with_llm(self, transaction_names):
        """
        Utilizes the language model to assign categories to a list of transaction names.

        Args:
            transaction_names (str): String containing concatenated transaction names.

        Returns:
            DataFrame: DataFrame with categorized transactions.
        """
        response = self.llm.invoke("You are a financial advisor. Can you assign an appropriate category to each transaction. Maintain the format: Date - Description - Amount - Category. Category names should be consistent and less than 3 words. " + transaction_names)
        response = response.split('\n')

        # Process response to extract categorized transactions
        categories_df = pd.DataFrame(columns=["Date", "Description", "Amount", "Category"])
        # Example processing, the actual logic might differ based on response format
        for line in response:
            if line.strip():  # Skip empty lines
                try:
                    date, description, amount, category = line.split(" - ")
                    categories_df = categories_df.append({"Date": date, "Description": description, "Amount": amount, "Category": category}, ignore_index=True)
                except ValueError:
                    # Handle cases where the line does not split correctly into four parts
                    continue
        
        return categories_df

    def _standardize_categories(self, categories_df):
        """
        Standardizes category names within the provided DataFrame.
        
        This method needs to be called within `categorize_transactions` to ensure category names are standardized
        after being assigned by the language model. It's implementation would depend on the specific categories
        and standardization rules required.

        Args:
            categories_df (DataFrame): DataFrame with transaction categories.
        """
        # Example standardization implementation is not provided as it depends on specific category names and rules.

    def aggregate_expenditures(self):
        """
        Creates an aggregated table of expenditures by category, including monthly and annual totals.

        Returns:
            DataFrame: A DataFrame with categories as rows and months plus an annual total as columns.
        """
        if self.df is None:
            raise ValueError("DataFrame is None. Ensure transaction data is loaded and categorized.")
        
        # Ensure 'Date' is in datetime format to extract month and year easily
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Year'] = self.df['Date'].dt.year
        
        # Group by Category, Year, and Month, then sum the Amounts
        monthly_expenditures = self.df.groupby(['Category', 'Year', 'Month']).agg({'Amount': 'sum'}).reset_index()
        
        # Pivot the table to have months and years as columns and categories as rows
        pivot_table = monthly_expenditures.pivot_table(index='Category', columns=['Year', 'Month'], values='Amount', aggfunc='sum', fill_value=0)
        
        # Add annual totals
        annual_totals = self.df.groupby(['Category', 'Year']).agg({'Amount': 'sum'}).reset_index()
        annual_totals_pivot = annual_totals.pivot_table(index='Category', columns='Year', values='Amount', aggfunc='sum', fill_value=0)
        annual_totals_pivot.columns = [f"Annual Total {col}" for col in annual_totals_pivot.columns]
        
        # Combine monthly and annual totals into one DataFrame
        aggregated_expenditures = pd.concat([pivot_table, annual_totals_pivot], axis=1)
        
        return aggregated_expenditures

    def export_categorized_transactions(self, aggregated_transactions, file_path):
        """
        Exports the DataFrame with categorized transactions and aggregated transactions to a CSV file.

        Args:
            file_path (str): The path to save the CSV file.
        """
        if self.df is not None:
            self.df.to_csv(file_path, index=False)
            print(f"Transaction data has been exported to {file_path}.")

            if aggregated_transactions is not None:
                aggregated_transactions.to_csv(file_path.replace(".csv", "_aggregated.csv"), index=True)
                print(f"Aggregated transactions have been exported to {file_path.replace('.csv', '_aggregated.csv')}.")
        else:
            raise ValueError("DataFrame is None. Ensure transaction data is loaded and processed.")


def main():
    # Initialize the finance analyzer with the specified model
    analyzer = FinanceAnalyzer(model_name="finance_gpt_llama2")
    
    # Read transaction data from a CSV file
    file_path = "/Users/mrinoyb2/git/FinanceGPT/data/transactions/amex_2023.csv"  # Update this path
    analyzer.read_transaction_data(file_path)
    
    # Clean and normalize transaction descriptions
    analyzer.clean_and_normalize(description_column='Description')
    
    # Categorize transactions using the language model
    analyzer.categorize_transactions()
    
    # Aggregate expenditures by category, including monthly and annual totals
    aggregated_expenditures = analyzer.aggregate_expenditures()
    print("Aggregated Expenditures:")
    print(aggregated_expenditures)
    
    # Export the aggregated transactions to a CSV file
    export_path = "/Users/mrinoyb2/git/FinanceGPT/data/llm_analyzed_transactions/categorized_transactions.csv"  # Update this path
    analyzer.export_categorized_transactions(aggregated_expenditures, export_path)
    
    print("Categorized and aggregated transactions have been exported successfully.")

if __name__ == "__main__":
    main()