# Intellex

\## Overview



Intellex is a knowledge-grounded AI chatbot designed for technical and engineering research. It follows a retrieval-first approach, using information from supplied documents and files as its primary knowledge source.



When the required information is not available in the internal knowledge base, Intellex can fall back to web research and use the retrieved information to formulate a contextual response.



\## Key Features



\- Knowledge-grounded question answering using user-provided documents

\- Retrieval-Augmented Generation (RAG) for technical information

\- Support for PDF, DOCX, TXT and structured knowledge sources

\- Vector-based document retrieval

\- Web-search fallback for information unavailable in the knowledge base

\- Context-aware response generation

\- Source-aware answers based on retrieved evidence

\- Designed for aerospace and engineering research workflows



\## How Intellex Works



Intellex follows a retrieval-first pipeline:



1\. The user submits a query.

2\. The query is processed and used to search the internal knowledge base.

3\. Relevant document sections are retrieved.

4\. The retrieved information is assembled into the model context.

5\. The AI generates an answer using the available evidence.

6\. If the internal knowledge base cannot answer the query, Intellex can perform web research.

7\. Retrieved web information is then used to generate a contextual response.



\## RAG Architecture



The document processing pipeline consists of:



\- Document ingestion and text extraction

\- Text cleaning and chunking

\- Metadata attachment

\- Embedding generation

\- Vector indexing

\- Similarity-based retrieval

\- Context construction

\- LLM-based response generation



The system treats the language model as the reasoning and language layer rather than the primary knowledge database.



\## Web Fallback



When relevant information cannot be found in the supplied knowledge base, Intellex can perform external web research.



This allows the system to handle queries involving information that may be missing from the local document collection, such as recent publications, specifications, standards and other up-to-date information.



\## Engineering Use Case



Intellex is designed to support engineering students and researchers by allowing technical documents to be queried conversationally.



For example, a user can ask a question about an aerospace engineering concept and have Intellex retrieve the relevant section from the supplied technical material before generating the response.



\## Project Goals



The main goals of Intellex are to:



\- Reduce the time required to search through technical documents

\- Ground AI responses in supplied engineering material

\- Reduce unsupported AI-generated answers

\- Provide contextual explanations from retrieved evidence

\- Extend the knowledge base through controlled web research

\- Provide a foundation for a larger engineering AI assistant



\## Future Development



Planned development includes:



\- Improved document ingestion

\- More advanced retrieval and ranking

\- Relevance thresholds for retrieved evidence

\- Improved web-source handling

\- Source citations and retrieval timestamps

\- Integration with engineering calculation tools

\- Natural-language access to engineering calculations

\- Tool orchestration between the chatbot and engineering APIs



\## Authors



\- Pranav Nandan L

\- Rahul Chava

