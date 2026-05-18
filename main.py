from dotenv import load_dotenv
import os

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
news_key = os.getenv("NEWS_API_KEY")
langchain_key = os.getenv("LANGCHAIN_API_KEY")
