# RAG Evaluation Test

Simple RAG system and evaluation implemented with `LangChain`. The RAG
uses a demo PDF of a fictional company's employee handbook. 
The evaluation pipeline uses other LLMs as graders and evaluates the
RAG's performance based
on the following metrics:

- **Correctness**: Is the generated answer correct?
- **Relevance**: Is the generated answer relevant to the question?
- **Groundedness**: Is the generated answer based on information
  that's actually in the document?
- **Relevance**: Are the retrieved document chunks relevant to the question?

## Usage
1. Add API keys to `.env`

Create a file called `.env` with the following:
```
GOOGLE_API_KEY=<API key goes here>
LANGSMITH_API_KEY=<API key goes here>
LANGSMITH_TRACING=true
```

2. Run the code

To run the chatbot: `uv run main.py`
To run the evaluator: `uv run evaluate.py`

## References
Based on [this tutorial](https://www.freecodecamp.org/news/rag-explained-simply-with-a-real-project/)
to build the RAG and [the LangChain docs](https://docs.langchain.com/langsmith/evaluate-rag-tutorial#configure-the-environment) for evaluation. 
