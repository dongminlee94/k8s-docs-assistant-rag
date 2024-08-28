"""Crawler."""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry
from requests.sessions import Session
from tqdm import tqdm


class K8sDocsCrawler:
    """Kubernetes Documentation Crawler.

    This class is used to crawl and download documentation from the Kubernetes website.
    """

    def __init__(self, url: str, target_subdomains: list[str]) -> None:
        """
        Initialize the K8sDocsCrawler.

        :param url: The base URL of the Kubernetes documentation.
        :param target_subdomains: List of target subdomains to process.
        """
        self._url = url
        self._base_url = self._get_base_url()
        self._target_subdomains = target_subdomains

        # Set up base directory for storing documentation
        self._base_dir = "data/docs"
        os.makedirs(self._base_dir, exist_ok=True)

        # Generate URL to file path mapping
        self._url_file_map = self._get_url_file_map()

    def _get_base_url(self) -> str:
        """
        Extract the base URL from the provided URL.

        :returns: The base URL.
        """
        splited_url = self._url.split("//")
        domain = splited_url[1].split("/")[0]

        return f"{splited_url[0]}//{domain}"

    def _get_url_file_map(self) -> dict[str, dict[str, str]]:
        """Map URLs to their corresponding file paths in the local storage.

        :returns: A dictionary with URLs as keys and their file paths as values.
        """
        url_file_map = {}

        def __get_url_file_map(input_dir: str) -> dict[str, dict[str, str]]:
            # Iterate over files in the input directory
            for entry in os.scandir(input_dir):
                if entry.is_dir():
                    __get_url_file_map(entry.path)
                elif entry.is_file():
                    # Load the JSON file and map URL to file path
                    with open(entry.path, "r", encoding="utf-8") as fp:
                        doc = json.load(fp)
                        url_file_map[doc[0]["url"]] = entry.path

            return url_file_map

        return __get_url_file_map(input_dir=self._base_dir)

    def _requests_retry_session(
        self,
        max_retries: int = 10,
        backoff_factor: int = 5,
        status_forcelist: list[int] = [400, 403, 429, 500, 502, 503, 504],
    ) -> Session:
        """
        Create a requests session with retry logic.

        :param max_retries: Maximum number of retries for failed requests.
        :param backoff_factor: Factor for exponential backoff strategy.
        :param status_forcelist: List of HTTP status codes to trigger a retry.
        :returns: A configured requests Session object.
        :note: If we do not receive the correct response when we send the request,
               retry as much as max_retries.
               We also use an "exponential backoff" strategy,
               where the delay is "backoff_factor * (2 ^ (retry count - 1))" after each failed retry.
        """
        # Configure retry strategy
        retry = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
        )

        # Create an HTTP adapter with the retry strategy
        adapter = HTTPAdapter(max_retries=retry)

        # Create and configure the session
        session = Session()
        session.mount(prefix="https://", adapter=adapter)

        return session

    def get_all_urls(self) -> list[str]:
        """
        Retrieve all documentation URLs from the main documentation page.

        :returns: A list of URLs pointing to documentation pages.

        Example:
            >>> crawler = K8sDocsCrawler(url="https://kubernetes.io/docs/home")
            >>> urls = crawler.get_all_urls()
            >>> print(urls)
            ['https://kubernetes.io/docs/concepts/', 'https://kubernetes.io/docs/tasks/', ...]
        """
        # Fetch the main documentation page
        response = self._requests_retry_session().get(self._url)
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract documentation URLs
        urls = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]

            # Check if the URL is within the target subdomains
            if href.startswith("/docs/") and href.split("/")[2] in self._target_subdomains:
                urls.append(self._base_url + href)

        return list(set(urls))

    def _generate_file_path(self, url: str, title: str) -> str:
        """Generate a file path for the documentation based on the URL and title.

        :param url: The documentation URL.
        :param title: The title of the documentation.
        :returns: A string representing the file path where the documentation JSON will be saved.
        """
        # Parse the URL to determine the directory structure
        splited_sub_url = url.split("/docs/")[1].split("/")
        output_dir = os.path.join(self._base_dir, splited_sub_url[0])
        os.makedirs(output_dir, exist_ok=True)

        # Create a unique file name using the title
        sub_path = "_".join(splited_sub_url[1:-1])
        suffix = title.replace(" ", "_").replace("/", "_") + ".json"

        # Combine path components into the final file path
        if sub_path:
            file_path = os.path.join(output_dir, sub_path + "_" + suffix)
        else:
            file_path = os.path.join(output_dir, suffix)

        return file_path

    def _create_doc(self, index: int, url: str) -> tuple[int, str, str | Exception, bool]:
        """
        Create a documentation JSON file from the given URL.

        :param index: The index of the URL in the list.
        :param url: The documentation URL.
        :returns: A tuple containing the index, URL, result, and success flag.
        :raises requests.exceptions.RequestException: If there is an issue with the HTTP request.
        :raises Exception: For other exceptions that may occur during processing.
        """
        result = None
        success = True

        try:
            # Fetch the documentation page
            response = self._requests_retry_session().get(url)
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract the title from the page
            title = soup.find("h1").text

            if not title:
                result = "No title"
                return index, url, result, success

            # Extract the main content from the page
            content = soup.find("div", {"class": "td-content"})

            if not content:
                result = "No content"
                return index, url, result, success

            # Parse content elements into a text format
            content_parts = []
            for element in content.descendants:
                if element.name in ["p", "ul", "ol", "pre", "code", "blockquote"]:
                    text = element.get_text(separator=" ", strip=True)
                    content_parts.append(text)
                elif element.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                    header_text = element.get_text(separator=" ", strip=True) + "\n"
                    content_parts.append(header_text)

            content = "\n".join(content_parts)

            # Check if the URL already has an existing document
            if url in self._url_file_map:
                old_file_path = self._url_file_map[url]

                # Compare old and new content
                with open(old_file_path, "r", encoding="utf-8") as fp:
                    old_content = json.load(fp)[0]["content"]

                if old_content == content:  # If content is unchanged, skip saving a new file
                    result = "Document already exists"
                    return index, url, result, success
                else:  # If content has changed, delete the old file
                    os.remove(old_file_path)
                    del self._url_file_map[url]

            # Generate file path and save new content as JSON
            file_path = self._generate_file_path(url=url, title=title)
            doc_data = [{"file_path": file_path, "title": title, "url": url, "content": content}]

            with open(file_path, "w", encoding="utf-8") as fp:
                json.dump(obj=doc_data, fp=fp, ensure_ascii=False, indent=4)

            # Update result and URL file map
            result = file_path
            self._url_file_map[url] = file_path
        except Exception as error:
            result = error
            success = False

        return index, url, result, success

    def create_all_docs(
        self, urls: list[str], max_workers: int = 200, verbose: bool = False
    ) -> list[dict[str, str | bool | None]]:
        """
        Create JSON documentation files for all provided URLs.

        :param urls: List of documentation URLs.
        :param max_workers: Maximum number of worker threads.
        :param verbose: Whether to display progress.
        :returns: A list of results, each containing the URL, result, and success flag.

        Example:
            >>> crawler = K8sDocsCrawler(url="https://kubernetes.io/docs/home")
            >>> urls = crawler.get_all_urls()
            >>> results = crawler.create_all_docs(urls=urls, verbose=True)
            >>> print(results)
            [{'url': 'https://kubernetes.io/docs/concepts/', 'result': None, 'success': True}, ...]
        """
        futures = []
        results = [None] * len(urls)

        # Set up the ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for index, url in enumerate(tqdm(urls) if verbose else urls):
                # Submit tasks to executor
                futures.append(executor.submit(self._create_doc, index=index, url=url))
                time.sleep(0.01)

            # Process completed futures as they are done
            for future in as_completed(futures):
                index, url, result, suceess = future.result()

                # Store the result in the results list
                item = {"url": url, "result": result, "suceess": suceess}
                results[index] = item

        return results
