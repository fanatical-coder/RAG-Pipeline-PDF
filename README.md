# RAG-Pipeline-PDF

The project contains a working RAG pipeline using the text scrapped out of PDFs.

Retrieval Augmented Generation is an AI framework that retrieves data from sources more like PDFs or internet databases. This retrieved data helps the Large Language Models to give out better answers.  

It is done via a series of events which includes- Data Scraping from the source -> Embedding the data, converting it to vector databases -> Retrieval, converting the user query into embeddings using the same model -> Searching the vector base to top-k matching results and Ranking them. -> Generation, Building a prompt, including the chunks and calling the LLM -> Answer.

It has intent detection for Image or Text retrieval. So, it can search for both Images and textual information.
