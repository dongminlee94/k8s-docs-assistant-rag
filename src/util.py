"""Utility."""


def make_chunks(data: str | list[int], length: int) -> list[str]:
    """
    Split the data into chunks of the specified length.

    :param data: The data to be split (either a string or a list of tokens).
    :param length: The length of each chunk.
    :returns: A list of chunks.

    Example:
        >>> data = "This is a sample text to be chunked."
        >>> chunks = DocsEmbedder.make_chunks(data=data, length=10)
        >>> print(chunks)
        ['This is a ', 'sample tex', 't to be c', 'hunked.']
    """
    return [data[i : i + length] for i in range(0, len(data), length)]
