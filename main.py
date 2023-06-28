import os
import env
import langchain

os.environ["OPENAI_API_KEY"] = env.API_KEY.openai

urls = [
    'https://www.mosaicml.com/blog/mpt-7b',
    'https://stability.ai/blog/stability-ai-launches-the-first-of-its-stablelm-suite-of-language-models',
    'https://lmsys.org/blog/2023-03-30-vicuna/'
]

from langchain.document_loaders import UnstructuredURLLoader
loaders = UnstructuredURLLoader(urls=urls)
data = loaders.load()

print(data)

# Text Splitter
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(separator='\n',
                                      chunk_size=1000,
                                      chunk_overlap=200)


docs = text_splitter.split_documents(data)

docs

len(docs)

import pickle
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

embeddings

# vectorStore_openAI = FAISS.from_documents(docs, embeddings)

# with open("faiss_store_openai.pkl", "wb") as f:
#     pickle.dump(vectorStore_openAI, f)

with open("faiss_store_openai.pkl", "rb") as f:
    VectorStore = pickle.load(f)

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI

llm=OpenAI(temperature=0, model_name='')

chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=VectorStore.as_retriever())

chain({"question": "How big is stableLM?"}, return_only_outputs=True)

chain({"question": "How good is Vicuna?"}, return_only_outputs=True)

chain({"question": "Which MPT-7B model is the bast one?"}, return_only_outputs=True)
