from bs4 import BeautifulSoup
import urllib.request

def fetch_article(url):
    try:
        # Send a GET request to the website
        response = urllib.request.urlopen(url)
        
        # Check if the request was successful
        if response.status == 200:
            # Get the website content (HTML)
            website_content = response.read().decode('utf-8')
            
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(website_content, 'html.parser')
            
            # Extract the article content
            article = soup.find('div', id='main-detail')
            if article:
                print("Article content fetched successfully!")
                return article.get_text(strip=True)  # Return the text content of the article
            else:
                print("Failed to find the article content.")
                return None
        else:
            print(f"Failed to fetch the website. Status code: {response.status}")
            return None

    except urllib.error.URLError as e:
        print(f"An error occurred: {e}")
        return None
