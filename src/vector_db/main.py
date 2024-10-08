"""Vector DB main module."""

import os

from crawler import K8sDocsCrawler
from embedder import DocsEmbedder

# Base URL for Kubernetes documentation
URL = "https://kubernetes.io/docs/home"

# Target subdomains to crawl within the Kubernetes documentation
# All subdomains: ["concepts", "contribute", "home", "reference", "setup", "tasks", "test", "tutorials"]
TARGET_SUBDOMAINS = ["home", "setup", "test"]

# OpenAI API key for accessing embedding services
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Embedding model configuration
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_MAX_TOKENS = 8192


def crawl_docs() -> None:
    """Crawl the Kubernetes documentation.

    This function initializes the K8sDocsCrawler and retrieves and processes
    all documentation URLs, saving the documents locally.
    """
    # Initialize the crawler with base URL and target subdomains
    crawler = K8sDocsCrawler(url=URL, target_subdomains=TARGET_SUBDOMAINS)

    # Retrieve all URLs to crawl
    urls = crawler.get_all_urls()

    # Create documentation files for all retrieved URLs
    results = crawler.create_all_docs(urls=urls, verbose=True)

    # Check for any failures in the crawling process
    failed = [result for result in results if not result["suceess"]]

    if failed:
        print(f"Crawling failed: {failed}")
        raise

    # Extract paths of newly created documents
    new_doc_paths = [result["result"] for result in results if ".json" in result["result"]]

    if new_doc_paths:
        print(
            f"The number of new document paths: {len(new_doc_paths)}\n"
            f"The list of new document paths: {new_doc_paths}\n"
        )


def embed_docs() -> None:
    """Embed the crawled documentation content.

    This function initializes the DocsEmbedder with the API key and target subdomains,
    and then processes the documentation to generate and save embeddings using the specified model.
    """
    # Initialize the embedder with the API key and target subdomains
    embedder = DocsEmbedder(api_key=OPENAI_API_KEY, target_subdomains=TARGET_SUBDOMAINS)

    # Generate and save embeddings for the crawled documentation
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
