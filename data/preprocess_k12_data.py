import requests
from bs4 import BeautifulSoup
import json
import logging
import time
from urllib.parse import urljoin
from requests.exceptions import RequestException, Timeout

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WIKIPEDIA_BASE_URL = "https://en.wikipedia.org"
SIMPLE_WIKIPEDIA_BASE_URL = "https://simple.wikipedia.org"
OUTPUT_FILE = "wikipedia_articles.jsonl"
PROGRESS_FILE = "progress.json"

# List of educational subjects to scrape
subjects = [
    "Biology",
    "Physics",
    "Chemistry",
    "History",
    "Geography"
]

# Configuration for retries
RETRIES = 3
BACKOFF_FACTOR = 2
REQUEST_DELAY = 2  # seconds
TIME_LIMIT = 30 * 60  # 30 minutes in seconds
SAVE_INTERVAL = 30

def fetch_article_content(url):
    for attempt in range(RETRIES):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                logger.error(f"Failed to retrieve the page. Status code: {response.status_code}")
                return None

            soup = BeautifulSoup(response.content, 'html.parser')
            title = soup.find('h1', {'id': 'firstHeading'}).text
            paragraphs = soup.find_all('p')

            content = {'title': title, 'paragraphs': [p.text for p in paragraphs if p.text.strip() != ""]}
            return content
        except (RequestException, Timeout) as e:
            logger.error(f"Request failed: {e}. Retrying in {BACKOFF_FACTOR ** attempt} seconds...")
            time.sleep(BACKOFF_FACTOR ** attempt)
    return None

def fetch_subject_articles(base_url, subject, start_index, max_articles=1000):
    subject_url = urljoin(base_url, f"/wiki/{subject}")
    response = requests.get(subject_url, timeout=10)
    if response.status_code != 200:
        logger.error(f"Failed to retrieve the subject page for {subject}. Status code: {response.status_code}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    content_div = soup.find('div', {'id': 'mw-content-text'})
    links = content_div.find_all('a', href=True, title=True)

    articles = []
    count = 0
    for i in range(start_index, len(links)):
        if count >= max_articles:
            break
        link = links[i]
        href = link['href']
        if href.startswith('/wiki/') and ':' not in href:
            full_url = urljoin(base_url, href)
            article_content = fetch_article_content(full_url)
            if article_content:
                articles.append(article_content)
                count += 1
                logger.info(f"Scraped article: {article_content['title']}")
                time.sleep(REQUEST_DELAY)
    
    return articles, start_index + count

def save_progress(progress):
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f)

def load_progress():
    try:
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def scrape_wikipedia_articles(base_url, max_articles=1000, save_interval=30):
    all_articles = []
    start_time = time.time()

    progress = load_progress()
    for subject in subjects:
        logger.info(f"Scraping articles related to {subject} from {base_url}")
        start_index = progress.get(subject, 0)
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > TIME_LIMIT:
                logger.info("Time limit reached. Saving current progress and stopping.")
                save_articles_to_file(all_articles, "partial_" + OUTPUT_FILE)
                save_progress(progress)
                return all_articles

            subject_articles, next_start_index = fetch_subject_articles(base_url, subject, start_index, save_interval)
            if not subject_articles:
                break
            all_articles.extend(subject_articles)
            save_articles_to_file(subject_articles, f"{subject}_articles.jsonl")
            progress[subject] = next_start_index
            save_progress(progress)
            start_index = next_start_index
            if len(subject_articles) < save_interval:
                break
            logger.info(f"Scraped {len(subject_articles)} articles for {subject} from {base_url}")

    save_progress(progress)
    return all_articles

def save_articles_to_file(articles, file_path):
    with open(file_path, 'a', encoding='utf-8') as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    logger.info("Starting to scrape Wikipedia articles...")
    wikipedia_articles = scrape_wikipedia_articles(WIKIPEDIA_BASE_URL)
    simple_wikipedia_articles = scrape_wikipedia_articles(SIMPLE_WIKIPEDIA_BASE_URL)

    logger.info("Saving articles to file...")
    save_articles_to_file(wikipedia_articles, "wikipedia_" + OUTPUT_FILE)
    save_articles_to_file(simple_wikipedia_articles, "simple_wikipedia_" + OUTPUT_FILE)

    logger.info("Scraping completed successfully.")
