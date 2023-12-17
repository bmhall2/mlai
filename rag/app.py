import bs4
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
import os

from langchain.document_loaders import WebBaseLoader

loader = DirectoryLoader(
    '/mnt/c/Learning/mlai/PdfTextTest/text/',
    glob="**/*.txt",
    show_progress=True,
    loader_cls=TextLoader
)

docs = loader.load()

# print(len(docs))
# print(len(docs[0].page_content))
# print(docs[0].page_content[:500])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# print(len(all_splits))
# print(len(all_splits[0].page_content))
# print(all_splits[10].metadata)

os.environ["OPENAI_API_KEY"] = ''
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# Done with Indexing

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

retrieved_docs = retriever.get_relevant_documents(
    "What is the name of the Partnership?"
)

# print(len(retrieved_docs))

# print(retrieved_docs[0].page_content)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

prompt = hub.pull("rlm/rag-prompt")

# print(
#     prompt.invoke(
#         {"context": "filler context", "question": "filler question"}
#     ).to_string()
# )

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

for chunk in rag_chain.stream("What is the name of the Partnership?"):
    print(chunk, end="", flush=True)