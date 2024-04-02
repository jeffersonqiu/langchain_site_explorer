from langchain_community.vectorstores import Chroma
from langchain_nomic import NomicEmbeddings
from langchain_community.document_loaders import DocusaurusLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from: https://python.langchain.com/docs/integrations/providers/nomic
embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")

#  from: https://python.langchain.com/docs/integrations/vectorstores/chroma
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# from: https://python.langchain.com/docs/integrations/document_loaders/docusaurus
loader = DocusaurusLoader(
    "https://python.langchain.com",
    filter_urls=[
        "https://python.langchain.com/docs/expression_language/"
    ],
)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=2000,
    chunk_overlap=20,
)

chunked_docs = text_splitter.split_documents(documents)

vectorstore.add_documents(chunked_docs)