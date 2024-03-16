import os
import re
import praw
import requests
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
from urllib.parse import urlparse
import ssl
import trafilatura
import json
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import time


def initialize_reddit():
    """
    Initializes and returns a Reddit instance using API credentials loaded from environment variables.
    
    Returns:
        praw.Reddit: Initialized Reddit instance.
    """
    load_dotenv()
    return praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"), 
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"), 
        user_agent = """Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36""")

def fetch_links_from_reddit_wiki(reddit, subreddit_name, wiki_page_title='index', visited_pages=set()):
    """
    Recursively fetches internal wiki links from a specified subreddit wiki page.

    Args:
        subreddit_name (str): Name of the subreddit.
        wiki_page_title (str): Title of the wiki page to start fetching from.
        visited_pages (set): Set of visited wiki page titles to prevent infinite recursion.

    Returns:
        list: List of unique internal wiki links.
    """
    internal_links = []
    try:
        wiki_page = reddit.subreddit(subreddit_name).wiki[wiki_page_title]
        content = wiki_page.content_md
        links = re.findall(r'\[.*?\]\((.*?)\)', content)
        for link in links:
            if f'reddit.com/r/{subreddit_name}/wiki/' in link:
                internal_link = link.split('/wiki/')[-1]
                if internal_link not in visited_pages:
                    visited_pages.add(internal_link)
                    internal_links.append(link)
                    internal_links.extend(fetch_links_from_reddit_wiki(subreddit_name, internal_link, visited_pages))
    except Exception as e:
        print(f"Error fetching page '{wiki_page_title}': {str(e)}")
    return list(set(internal_links))

def get_sitemap(url):
    """
    Fetches and parses the sitemap XML from a given URL.

    Args:
        url (str): URL of the sitemap to fetch.

    Returns:
        BeautifulSoup: Parsed sitemap XML.
    """
    req = Request(url=url, headers={"User-Agent": "Mozilla/5.0"})
    response = urlopen(req)
    xml = BeautifulSoup(response, "lxml-xml", from_encoding=response.info().get_param("charset"))
    return xml

def sitemap_to_dataframe(xml, name=None, verbose=False):
    """
    Converts a sitemap XML to a pandas DataFrame.

    Args:
        xml (BeautifulSoup): Parsed sitemap XML.
        name (str, optional): Name to assign to the sitemap. Defaults to None.
        verbose (bool, optional): If True, prints each row as it's processed. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame containing sitemap information.
    """
    df = pd.DataFrame(columns=["loc", "changefreq", "priority", "domain", "sitemap_name"])
    urls = xml.find_all("url")
    for url in urls:
        loc = url.find("loc").text if url.find("loc") else ""
        parsed_uri = urlparse(loc)
        domain = f"{parsed_uri.netloc}"
        changefreq = url.find("changefreq").text if url.find("changefreq") else ""
        priority = url.find("priority").text if url.find("priority") else ""
        sitemap_name = name if name else ""
        row = {"domain": domain, "loc": loc, "changefreq": changefreq, "priority": priority, "sitemap_name": sitemap_name}
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df

def extract_relevant_text_from_zerodha_urls(url):
    downloaded_url = trafilatura.fetch_url(url)
    extracted = trafilatura.extract(
        downloaded_url, 
        output_format="json", 
        with_metadata=True, 
        include_comments = False,
        date_extraction_params={"extensive_search": True, "original_date": True}
    )
    json_output = json.loads(extracted)
    return json_output["text"]

def extract_relevant_text_from_reddit_urls(url, retries=5, backoff_factor=1.0):
    """
    Fetches and extracts main content from a given URL with retries.

    Args:
        url (str): The URL to extract text from.
        retries (int): Number of retries on fetch failure.
        backoff_factor (float): Factor by which to increase delay between retries.

    Returns:
        str: Extracted text content or error message.
    """
    for attempt in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                main_content = soup.find('main') or soup.find('article') or soup.find('div', role='main')
                if not main_content:
                    main_content = soup.body
                text = ' '.join(t.get_text(separator='\n', strip=True) for t in main_content.find_all(recursive=False) if t.name not in ['script', 'style'])
                if "An unknown error occurred" not in text:
                    # return text without the first 6 lines (usually contains metadata)
                    # remove any lines with just one word (usually contains metadata)
                    return '\n'.join([line for line in text.split('\n')[6:] if len(line.split()) > 1])
                else:
                    raise ValueError("Content indicates an error page")
            else:
                raise ValueError(f"HTTP error {response.status_code}")
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(backoff_factor * (2 ** attempt))
    return "Failed to retrieve content after several attempts"

def main():
    """
    Main function to orchestrate the fetching of wiki links from Reddit, processing of sitemaps,
    and extraction of content from URLs.
    """
    reddit = initialize_reddit()
    
    # Set up SSL context for trafilatura
    ssl._create_default_https_context = ssl._create_stdlib_context

    subreddit_name = 'personalfinance'
    # Make sure to update function definitions to accept 'reddit' as a parameter and adjust function calls accordingly
    reddit_wiki_links = fetch_links_from_reddit_wiki(reddit, subreddit_name)
    print(f"Collected {len(reddit_wiki_links)} unique internal wiki links from Reddit.")

    sitemap_url = "https://zerodha.com/varsity/chapter-sitemap2.xml"
    xml = get_sitemap(sitemap_url)
    df = sitemap_to_dataframe(xml, verbose=True)
    zerodha_urls = df["loc"].to_numpy()
    zerodha_urls = [url for url in zerodha_urls if "%" not in url]
    print(f"Collected {len(zerodha_urls)} unique URLs from Zerodha sitemap.")

    save_dir = "/Users/mrinoyb2/git/FinanceGPT/data/test_folder" 
    os.makedirs(save_dir, exist_ok=True)

    print("Working on scraping zerodha urls....")
    with open(os.path.join(save_dir, "zerodha_content.txt"), "w") as f:
        for url in tqdm(zerodha_urls[1:]):
            topic = url.split("/")[-2]
            if "hindi" in topic or topic in ["the-vegetable-list", "bonus-share-vs-stock-split", "getting-started-2"]:
                continue
            text = extract_relevant_text_from_zerodha_urls(url=url)
            text = text.lower()
            text = text.replace("key takeaways from this chapter", "")
            text = text.replace("we recommend reading this chapter on varsity to learn more and understand the concepts in-depth.", "")
            text = text.replace("varsity", "")
            f.writelines(topic + "\n")
            f.writelines(text  + "\n###\n")

    print("Working on scraping reddit wiki links....")
    for url in tqdm(reddit_wiki_links):
        text = extract_relevant_text_from_reddit_urls(url)
        filename = re.sub(r'\W+', '', url.split('/')[-1]) + '.txt'
        filepath = os.path.join(save_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Saved content from {url} to {filename}")

    all_urls = list(set(reddit_wiki_links + list(zerodha_urls)))
    print(f"Finished Scraping! Total unique URLs scraped: {len(all_urls)}")

if __name__ == "__main__":
    main()