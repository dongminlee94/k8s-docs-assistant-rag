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

- **Up-to-date Information**: Kumi provides highly accurate and reliable responses by relying on the latest Kubernetes document data.
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

설치 방법은 아래와 같습니다.

```bash
# user
$ make setup

# developer
$ make setup-dev
```

## Preparation

Vector DB를 만들고, Real-time Chat을 하기 위해서는 OpenAI API Key가 필요합니다.

[OpenAI API Key](https://platform.openai.com/api-keys)를 통해 API Key를 생성합니다.

이 저장소에서 `.env` 파일을 생성하고, `.env` 파일 안에 아래와 같이 입력합니다.

```env
OPENAI_API_KEY=xxxxx
```

## How To Chat

### 1. Vector DB Setup

Vector DB를 만들기 위해 [Vector DB main module](src/vector_db/main.py)에서 신경써야할 변수는 3가지입니다.

```python
TARGET_SUBDIRS = ["concepts", "contribute", "home", "setup", "tasks", "test", "tutorials"]

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_MAX_TOKENS = 8192
```

- Target subdirs: target subdirs 리스트를 수정하여 원하는 subdir의 문서 데이터를 추출할 수 있습니다. **모든 subdir을 넣게 되면 약 800개의 문서 데이터를 OpenAI API를 이용하여 임베딩 해야하기 때문에 비용이 조금 발생할 수도 있습니다.**
- Embedding Model: [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings/embedding-models) 문서를 참고하여 원하는 임베딩 모델을 기입합니다.
- Embedding Max Tokens: [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings/embedding-models)를 참고하여 해당 임베딩 모델의 max tokens를 기입합니다. 일반적으로 max input에 적힌 숫자에 플러스 1을 하면 됩니다.

변경이 끝났으면 코드를 반드시 저장해주세요.

### 2. API Setup

[API main module](src/api/main.py)에서 신경써야할 변수는 4가지입니다.

```python
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_MAX_TOKENS = 8192
COMPLETION_MODEL = "gpt-4o-mini"
COMPLETION_CONTEXT_WINDOW = 128000
```

- Embedding Model: Vector DB Setup과 동일합니다.
- Embedding Max Tokens: Vector DB Setup과 동일합니다.
- Completion Model: [OpenAI Models](https://platform.openai.com/docs/models) 문서를 참고하여 원하는 Chat Completion 모델을 기입합니다.
- Completion Context Window: [OpenAI Models](https://platform.openai.com/docs/models) 문서를 참고하여 해당 모델의 context window를 기입합니다.

### 3. Real-time Chat with Kumi

```bash
$ make chat
```

```bash
$ make chat-end
```
