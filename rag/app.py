import argparse
from langchain import hub
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, type=str)
    parser.add_argument("--fileId", required=True, type=int)

    args = parser.parse_args()
    query = args.query
    fileId = args.fileId

    open_api_key = ''

    client = Chroma(
        collection_name="rag",
        persist_directory="./chroma",
        embedding_function = OpenAIEmbeddings(
            api_key=open_api_key
        )
    )

    retriever = client.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 6,
            "filter" : {'fileId':fileId}
        })

    retrieved_docs = retriever.get_relevant_documents(query)

    # print(retrieved_docs[0].metadata)

    if len(retrieved_docs) > 0:

        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=open_api_key)

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

        for chunk in rag_chain.stream(query):
            print(chunk, end="", flush=True)
    else:
        print("No local results")