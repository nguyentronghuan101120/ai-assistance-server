# Chat bot with terminal interface

import os
import openai
from dotenv import load_dotenv
from fetch_article import fetch_article
import re

# Load environment variables
load_dotenv()

# Initialize OpenAI API client
api_key = os.getenv("together_api_key")
client = openai.OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=api_key,
)

def get_chat_completion(prompt):
    """
    Sends a prompt to the OpenAI API and returns the response.
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            stream=False,
            max_completion_tokens=500,  # Adjust token limit for summaries
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating completion: {e}"

def summarize_article(url):
    """
    Fetches the article content from the given URL and summarizes it.
    """
    print("\nFetching article content...")
    article_content = fetch_article(url)
    if article_content:
        print("\nSummarizing article content...")
        summary_prompt = f"Summarize the following article in its original language, maintaining the tone, context, and key points. Ensure the summary is concise but retains all critical information and does not change the meaning of the original text. Do not translate or alter the language of the article:\n\n{article_content}"
        summary = get_chat_completion(summary_prompt)
        return summary
    else:
        return "Failed to fetch the article content."

def extract_url(text):
    """
    Extracts the first URL found in the given text.
    """
    url_pattern = re.compile(r'https?://\S+')
    match = url_pattern.search(text)
    return match.group(0) if match else None

def main():
    """
    Main function to run the chatbot loop.
    """
    print("Welcome! Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "exit":
            print("\nGoodbye!")
            break

        # Extract URL from the user input
        article_url = extract_url(user_input)
        if article_url:
            summary = summarize_article(article_url)
            print("\nArticle Summary:\n", summary)
        else:
            # Regular chat response
            response = get_chat_completion(user_input)
            print("\nBot:", response)

if __name__ == "__main__":
    main()
