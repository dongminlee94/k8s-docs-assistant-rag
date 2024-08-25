# Kumi: Kubernetes Documentation Assistant with RAG

## Introduction

**Kumi** is a documentation assistant designed to help you efficiently explore and understand Kubernetes documentation using RAG (Retrieval-Augmented Generation) technology.
With approximately 800 official Kubernetes documents, finding the right information can be time-consuming and challenging.
Kumi was developed to address this issue.

As of August 2024, the Kubernetes official documentation consists of various subdomains, each containing a different number of documents.
Below is a breakdown of the structure and the number of documents available in each subdomain:

- https://kubernetes.io/docs/concepts: 154 documents
- https://kubernetes.io/docs/contribute: 33 documents
- https://kubernetes.io/docs/home: 1 documents
- https://kubernetes.io/docs/reference: 309 documents
- https://kubernetes.io/docs/setup: 24 documents
- https://kubernetes.io/docs/tasks: 209 documents
- https://kubernetes.io/docs/test: 1 documents
- https://kubernetes.io/docs/tutorials: 44 documents

Given the vast amount of documentation, using RAG offers several benefits to make navigating and understanding these documents more efficient:

- **Up-to-date Information**: Kumi provides highly accurate and reliable responses using the latest Kubernetes document data.
- **Efficiency**: Kumi significantly reduces the time required to find relevant information, offering quick and precise responses to your queries.
- **Scalability**: Kumi enhances access to a broad range of Kubernetes documentation, supporting a deeper technical understanding.

Currently, Kumi's responses include some Korean, so it's recommended to ask questions in **Korean** for the best results.

## How It Works

Kumi operates through two main processes:

### Vector DB Setup

1. **Data Collection**: All Kubernetes documents are crawled to collect document data.
2. **Embedding**: The collected document data is embedded using an Embedding model and stored in a Vector DB.

### Real-time Chat

1. **Processing User Input**: When a user inputs a query, it is embedded using the Embedding model.
2. **Relevant Document Search**: The system searches the Vector DB for the most relevant content based on the user's input.
3. **Answer Generation**: Using the retrieved content, a Completion model generates an optimal response for the user.

## Installation

이 저장소를 사용하기 위한 설치 방법은 아래와 같습니다.

```bash
# For general users
$ make setup

# For developers
$ make setup-dev
```

## Prerequisites

To set up the Vector DB and enable Real-time Chat, you'll need an OpenAI API Key.

Generate an API Key from [OpenAI API Key](https://platform.openai.com/api-keys).

Create a .env file in this repository and enter the following line with your API Key:

```env
OPENAI_API_KEY=xxxxx
```

## How To Chat

### 1. Vector DB Setup

To set up the Vector DB, check the following three variables in the [Vector DB main module](src/vector_db/main.py).

```python
# Default variables
TARGET_SUBDOMAINS = ["concepts", "contribute", "home", "reference", "setup", "tasks", "test", "tutorials"]

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_MAX_TOKENS = 8192
```

- **Target subdomains**: Modify the list of target subdomains to extract document data from the desired subdomains. **Note: Including all subdomains will result in embedding approximately 800 documents using the OpenAI API, which could incur additional costs.**
- **Embedding Model**: Refer to the [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings/embedding-models) and specify the embedding model you want to use.
- **Embedding Max Tokens**: Check the [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings/embedding-models) to find the maximum tokens for the chosen embedding model. Generally, it is the number listed under max input plus one.

**Ensure you save your changes to the code after modifying these variables.**

### 2. API Setup

In the [API main module](src/api/main.py), ensure you check the following four variables:

```python
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_MAX_TOKENS = 8192
COMPLETION_MODEL = "gpt-4o-mini"
COMPLETION_CONTEXT_WINDOW = 128000
```

- **Embedding Model**: Same as in the Vector DB Setup.
- **Embedding Max Tokens**: Same as in the Vector DB Setup.
- **Completion Model**: Refer to the [OpenAI Models](https://platform.openai.com/docs/models) to choose your preferred Chat Completion model.
- **Completion Context Window**: Refer to the [OpenAI Models](https://platform.openai.com/docs/models) to find the context window for the selected model.

### 3. Real-time Chat with Kumi

To start a chat session with Kumi, enter the following command:

```bash
$ make chat
```

To end the chat session with Kumi, use the following command:

```bash
$ make chat-end
```

Then, open your web browser and go to localhost:9000 to start chatting with Kumi!

Enjoy your conversation with Kumi and happy exploring!
Thank you for using Kumi.
