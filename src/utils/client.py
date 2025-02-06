import os
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI API client
client = openai.OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="none",
)
