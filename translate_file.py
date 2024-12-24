import os
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI API client
api_key = os.getenv("together_api_key")
client = openai.OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=api_key,
)

def translate_text(text, target_language):
    """
    Translates the given text to the target language using OpenAI API.
    """
    prompt = f"Translate the following text to {target_language}:\n\n{text}"
    try:
        translation = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            stream=False,
            max_completion_tokens=1000,
        )
        return translation.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating translation: {e}"

def translate_file(input_file, output_file, target_language):
    """
    Reads the content of the input file, translates it, and writes the translated content to the output file.
    """
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()
    
    translated_content = translate_text(content, target_language)
    
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(translated_content)

if __name__ == "__main__":
    input_file = "input_file.txt"
    output_file = "output_file.txt"
    target_language = "Spanish"  # Specify the target language
    translate_file(input_file, output_file, target_language)
