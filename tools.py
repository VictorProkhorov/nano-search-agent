from langchain_community.retrievers import BM25Retriever
from langchain_community.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from smolagents import Tool
from ddgs import DDGS

import ujson as json

class PartyPlanningRetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve relevant articles from Wikipedia."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be a query related to answering the question. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        
        # Convert dataset entries to Document objects with metadata
        source_docs = [
        Document(page_content=doc["contents"], metadata={"id": doc["id"]})
            for doc in docs]

        # Split documents into smaller chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Characters per chunk
        chunk_overlap=50,  # Overlap between chunks to maintain context
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", " ", ""],  # Priority order for splitting
        )
        
        wiki_docs = text_splitter.split_documents(source_docs)
        
        self.retriever = BM25Retriever.from_documents(
            wiki_docs, k=5  # Retrieve the top 5 documents
        )

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"
        print('q', query)
        docs = self.retriever.invoke(
            query,
        )
        return "\nRetrieved Wikipedia documents:\n" + "".join(
            [
                f"\n\nWikipedia document {str(i)}\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )

def load_wiki(path: str):
    wiki = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line_id, line in enumerate(f):
            line = line.strip()
            if not line.startswith('{'):
                continue  # skip tar header lines
            try:
                wiki.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if line_id == 500000:
               break
    print(f"Loaded {len(wiki)} records")
    return wiki


RETRIVAL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "retriever",
        "description": "Uses semantic search to retrieve relevant articles from Wikipedia.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to perform. This should be a query related to answering the question. Use the affirmative form rather than a question.",
                }
            },
            "required": ["query"],
        },
    },
}


def ddg_search(query: str, k: int = 5) -> str:
    with DDGS() as ddgs:
        hits = list(ddgs.text(query, safesearch="moderate", max_results=k))
    if not hits:
        return "No results found."
    return "\n".join(
        f"{i+1}. {h['title']} - {h['body']} ({h['href']})"
        for i, h in enumerate(hits)
    )

SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "DuckDuckGo web search. Use it when you need external knowledge.",
        "parameters": {
            "type": "object",
            "properties": {
                "query_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "One or more fully-formed semantic search queries.",
                }
            },
            "required": ["query_list"],
        },
    },
}


def test_tool():
    #with open('./data/wiki-18.jsonl', 'rb') as f:
    #    print(f.read(20))
    #exit()
    wiki_docs = load_wiki('./data/wiki-18.jsonl')
    #print(wiki_docs[:4])
    #print(wiki_docs[0].keys())
    tool = PartyPlanningRetrieverTool(wiki_docs)
    result = tool('hell')
    print(result)
    result = tool('suez')
    print(result)
    
    return


if __name__ == '__main__':
    test_tool()