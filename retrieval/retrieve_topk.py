# retrieval.py
from langchain_community.tools.bearly.tool import head_file
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Set your OpenAI API Key here directly


# Location of the saved FAISS vector store
VECTOR_STORE_DIR =  r"C:\Users\msdha\PyCharmMiscProject\new\embeddings\cache"

def load_vector_store():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
    return db

def retrieve_top_k(query: str, k: int = 10):
    db = load_vector_store()
    results = db.similarity_search(query, k=k)  # remove this by MMR(MAXIMUM RELEVANCE RETRIEVAL)
    return results

if __name__ == "__main__":
    query = input("Enter your query: ")
    docs = retrieve_top_k(query)
    # print(docs[0])
    print(repr(docs[0]))
    print("ID:", docs[0].id)
    print(type(docs))
    print("\nTop relevant chunks:")
    for i, doc in enumerate(docs, start=1):
        print(f"\n--- Chunk {i} ---")
        print(f"Source: {doc.metadata.get('source')}")
        print(doc.page_content)

# retriever = vectorstore.asretrieve(search_kwargs={"query": query, "k": 5})
# query = retriever.invoke(query)
# results = db.similarity_search(query, k=5) both produce same answer



class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True

        return False

    slow = head



