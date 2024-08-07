"""Vector DB."""

import os

from core.crawler import K8sDocsCrawler
from core.embedder import DocsEmbedder
from core.summarizer import DocsSummarizer

URL = "https://kubernetes.io/docs/home"

TARGET_SUBDIRS = ["home", "setup", "test"]
SUMMARIZER_PROMPT_NAME = "summary"
SUMMARIZER_MODEL = "gpt-4o-mini"
SUMMARIZER_CONTEXT_WINDOW = 128000

EMBEDDER_MODEL = "text-embedding-3-large"
EMBEDDER_MAX_TOKENS = 8192


def get_api_key() -> str:
    """Retrieve the API key from the environment file.

    :returns: The API key as a string.
    """
    with open(os.path.join(os.path.dirname(__file__), "../..", "env/api_key.env"), "r") as file:
        api_key = file.read().strip()

    return api_key


def crawl_docs() -> None:
    """Crawl the Kubernetes documentation.

    This function initializes the K8sDocsCrawler and retrieves and processes
    all documentation URLs, saving the documents locally.
    """
    crawler = K8sDocsCrawler(url=URL)

    urls = crawler.get_all_urls()
    results = crawler.create_all_docs(urls=urls, verbose=True)

    failed = [result for result in results if not result["suceess"]]

    if failed:
        print(f"crawling failed: {failed}")
        raise


def summarize_docs() -> None:
    """Summarize the crawled documentation.

    This function initializes the DocsSummarizer, reads the crawled documents,
    and generates summaries for each document.
    """
    api_key = get_api_key()

    summarizer = DocsSummarizer(
        target_subdirs=TARGET_SUBDIRS,
        api_key=api_key,
        prompt_name=SUMMARIZER_PROMPT_NAME,
    )
    summarizer.summarize_docs(model=SUMMARIZER_MODEL, context_window=SUMMARIZER_CONTEXT_WINDOW, verbose=True)


def embed_docs() -> None:
    """Embed the summarized documentation.

    This function initializes the DocsEmbedder, reads the summarized documents,
    and generates embeddings for each document.
    """
    api_key = get_api_key()

    embedder = DocsEmbedder(api_key=api_key)
    embedder.embed_docs(model=EMBEDDER_MODEL, max_tokens=EMBEDDER_MAX_TOKENS, verbose=True)


def main() -> None:
    """Main function to run the documentation processing pipeline.

    This function orchestrates the crawling, summarizing, and embedding of
    the Kubernetes documentation.
    """
    print("Start to create a vector DB\n")

    print("Step 1: Crawl the documentations")
    crawl_docs()
    print("The Crawling process completed\n")

    print("Step 2: Summarize the documentations")
    summarize_docs()
    print("The Summarizing process completed\n")

    print("Step 3: Embed the documentations")
    embed_docs()
    print("The Embedding process completed\n")

    print("The vector DB creation process completed")


if __name__ == "__main__":
    main()
