"""Vector DB main function."""

import os

from core.crawler import K8sDocsCrawler
from core.embedder import DocsEmbedder

URL = "https://kubernetes.io/docs/home"

# Subdirs: "concepts", "contribute", "home", "reference", "setup", "tasks", "test", "tutorials"
TARGET_SUBDIRS = ["home", "setup", "test"]

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_MAX_TOKENS = 8192


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
    crawler = K8sDocsCrawler(url=URL, target_subdirs=TARGET_SUBDIRS)

    urls = crawler.get_all_urls()
    results = crawler.create_all_docs(urls=urls, verbose=True)

    failed = [result for result in results if not result["suceess"]]

    if failed:
        print(f"crawling failed: {failed}")
        raise


def embed_docs() -> None:
    """Embed the crawled documentation content.

    This function initializes the DocsEmbedder with the API key and target subdirectories,
    and then processes the documentation to generate and save embeddings using the specified model.
    """
    api_key = get_api_key()

    embedder = DocsEmbedder(api_key=api_key, target_subdirs=TARGET_SUBDIRS)
    embedder.embed_docs(model=EMBEDDING_MODEL, max_tokens=EMBEDDING_MAX_TOKENS, verbose=True)


def main() -> None:
    """Main function to run the documentation processing pipeline.

    This function orchestrates the crawling and embedding of the Kubernetes documentation.
    """
    print("Start to create a vector DB\n")

    print("Step 1: Crawl the documentation")
    crawl_docs()
    print("The Crawling process completed\n")

    print("Step 2: Embed the documentation")
    embed_docs()
    print("The Embedding process completed\n")

    print("The vector DB creation process completed")


if __name__ == "__main__":
    main()
