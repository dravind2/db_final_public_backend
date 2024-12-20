import os
from typing import List, Dict
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain


class MagicalRAG:
    def __init__(self, openai_api_key: str = None):
        """
        Initializes our RAG system for magical knowledge.

        Args:
            openai_api_key: API key for OpenAI services. If None, will attempt to load from environment.
        """
        # Load environment variables from .env file
        load_dotenv()

        # Get API key from parameter or environment
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Please provide it as a parameter "
                "or set the OPENAI_API_KEY environment variable."
            )

        # Set up our language model and embeddings with the new package
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            openai_api_key=self.api_key,
            streaming=False,
        )

        # Initialize vector store and QA chain as none at first (later edit: don't really use history because we don't store cookies or have login-- no authentication)
        self.vector_store = None
        self.qa_chain = None

        # Keep track of conversation history (not really used)
        self.chat_history = []

        # Prompt
        self.custom_prompt = PromptTemplate(
            template="""Given the following conversation and a follow up question, provide a detailed response that includes:
            1. Specific magical solutions (spells and/or potions) that would be most helpful
            2. How to properly use these magical solutions
            3. Any important warnings or considerations
            4. If relevant, suggest combinations of spells and potions that work well together

            Chat History:
            {chat_history}

            Follow Up Input: {question}

            Base your response only on the magical knowledge provided in the context.
            If multiple solutions exist, explain the tradeoffs between them.

            Context: {context}

            Answer:""",
            input_variables=["chat_history", "question", "context"],
        )

    def process_magical_data(
        self, spells_df: pd.DataFrame, potions_df: pd.DataFrame
    ) -> List[str]:
        """
        Converts our CSV data into detailed text documents that capture all important
        information about spells and potions in a natural language format.

        Args:
            spells_df: DataFrame containing spell information
            potions_df: DataFrame containing potion information

        Returns:
            List of documents describing magical items
        """
        documents = []

        # Process spells into detailed descriptions
        for _, spell in spells_df.iterrows():
            description = f"""
            Spell Name: {spell['name']}
            This is a {spell['category']} spell.
            Effect: {spell['effect']}
            """

            if pd.notna(spell["incantation"]):
                description += f"The incantation used is: {spell['incantation']}\n"

            if pd.notna(spell["light"]):
                description += f"When cast, it produces {spell['light']} light.\n"

            documents.append(description)

        # Process potions into detailed descriptions
        for _, potion in potions_df.iterrows():
            description = f"""
            Potion Name: {potion['name']}
            This potion has the following effect: {potion['effect']}
            """

            if pd.notna(potion["characteristics"]):
                description += f"Characteristics: {potion['characteristics']}\n"

            if pd.notna(potion["ingredients"]):
                description += f"Key ingredients include: {potion['ingredients']}\n"

            if pd.notna(potion["difficulty"]):
                description += f"Difficulty level: {potion['difficulty']}\n"

            documents.append(description)

        return documents

    # Accept spells and potions dataframes as arguments, then create the knowledge base
    def create_knowledge_base(self, spells_df, potions_df):
        """
        Creates our magical knowledge base by processing CSV files and storing
        them in a vector database for efficient retrieval.

        Args:
            spells_path: Path to spells CSV file
            potions_path: Path to potions CSV file
        """

        # Convert data into documents
        documents = self.process_magical_data(spells_df, potions_df)

        # Split documents into smaller chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ". ", " "]
        )

        texts = text_splitter.create_documents(documents)

        # Create our vector store
        self.vector_store = FAISS.from_documents(texts, self.embeddings)

        # Create our question-answering chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 5}  # Retrieve top 5 most relevant chunks (k=5 most similar)
            ),
            return_source_documents=True,
        )

    # Accept a query and return a response
    def get_magical_advice(self, query: str) -> Dict:
        """
        Provides magical advice and recommendations based on the query.

        Args:
            query: User's question or scenario

        Returns:
            Dictionary containing the response and relevant source information
        """
        # Create a more specific prompt to guide the LLM
        enhanced_query = f"""
        Based on the following scenario or question: {query}
        
        Please provide a detailed response that includes:
        1. Specific magical solutions (spells and/or potions) that would be most helpful
        2. How to properly use these magical solutions
        3. Any important warnings or considerations
        4. If relevant, suggest combinations of spells and potions that work well together
        5. Can you provide a pros and cons to using each spell/potion and rank them in terms of ease of use and effectiveness?
        
        Base your response only on the magical knowledge provided in the context.
        If multiple solutions exist, explain the tradeoffs between them.
        At the end, wish the wizard or witch good luck on their magical journey and to use magic responsibly!
        """

        # Get response from our QA chain
        response = self.qa_chain.invoke(
            {"question": enhanced_query, "chat_history": self.chat_history}
        )

        # Update chat history (not really used)
        self.chat_history.append((query, response["answer"]))

        # Format the response
        return {
            "answer": response["answer"],
            "sources": [doc.page_content for doc in response["source_documents"]],
            "chat_history": self.chat_history,
        }

    # Accept a spell or potion name and return similar magical items
    def get_similar_magic(self, item_name: str) -> List[str]:
        """
        Finds magically similar items to the one specified.

        Args:
            item_name: Name of the spell or potion to find similar items for

        Returns:
            List of similar magical items with explanations
        """
        # Create a query to find similar items
        query = f"Find magical items similar to {item_name} in terms of effects or uses"
        docs = self.vector_store.similarity_search(query, k=3)

        return [doc.page_content for doc in docs]

# Use python connector to connect to MySQL database and retrieve spells and potions data
def get_data_from_db():
    """Get spells and potions data from MySQL database."""
    try:
        connection = mysql.connector.connect(
            host=os.getenv("HOST"),
            database=os.getenv("DATABASE"),
            user=os.getenv("USERNAME"),
            password=os.getenv("PASSWORD"),
        )

        # Get spells
        spells_query = "SELECT * FROM spells"
        spells_df = pd.read_sql(spells_query, connection)

        # Get potions
        potions_query = "SELECT * FROM potions"
        potions_df = pd.read_sql(potions_query, connection)

        return spells_df, potions_df

    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        raise
    finally:
        if "connection" in locals() and connection.is_connected():
            connection.close()

# Main function which is called by the FastAPI endpoint
def main(single_query: str = None):

    """
    Main function to demonstrate the usage of MagicalRAG.
    """
    try:
        # Initialize our RAG system
        rag = MagicalRAG() 

        spells_df, potions_df = get_data_from_db()

        # Verify that the data files exist
        # if not spells_df.exists() or not potions_df.exists():
        #     raise FileNotFoundError(
        #         "Data files not found."
        #     )

        # Create the knowledge base
        rag.create_knowledge_base(spells_df=(spells_df), potions_df=(potions_df))

        # Example queries
        # queries = [
        #     "I need to defend against dark magic. What spells and potions would you recommend?",
        #     "How can I heal someone who was injured in a magical duel?",
        #     "What are some good combinations of spells and potions for stealth missions?",
        # ]

        response = rag.get_magical_advice(single_query)
        return {
            "answer": response["answer"],
            "sources": [source.strip() for source in response["sources"]]
        }

    except Exception as e:
        print(f"An error occurred: {str(e)}")
