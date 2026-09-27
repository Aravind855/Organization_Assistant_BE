import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def get_llm():
    """
    Returns an instance of ChatGoogleGenerativeAI using GEMINI_API_KEY from environment.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
    
    if not api_key or api_key.strip() == "" or api_key == "your_gemini_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY is missing or unconfigured. Please add your key to Backend/.env"
        )
        
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
