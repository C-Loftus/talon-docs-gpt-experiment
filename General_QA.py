import os, env
import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
import pickle
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
import streamlit as st

os.environ["OPENAI_API_KEY"] = env.API_KEY.openai
text_splitter = CharacterTextSplitter(separator='\n',
                                    chunk_size=1000,
                                    chunk_overlap=200)

st.title('🦜 Talon GPT: Unofficial Talon Help')
"Note: Consult the Talon Slack if you are stuck or need the most up-to-date information."

@st.cache_data
def get_urls():
    sitemap = "https://talon.wiki/sitemap.xml"
    wunder = requests.get(sitemap)
    parcala = BeautifulSoup(wunder.content, "xml")

    urls_from_xml = ["https://talonvoice.com/docs/index.html", "https://github.com/knausj85/knausj_talon"]

    loc_tags = parcala.find_all('loc')

    for loc in loc_tags:
        urls_from_xml.append(loc.get_text()) 
    loaders = UnstructuredURLLoader(urls=urls_from_xml)
    data = loaders.load()
    return data

with st.spinner('Starting up application... '):
    data = get_urls()
    docs = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings()
    @st.cache_data
    def init_db():
        return FAISS.from_documents(docs, embeddings)
    vectorStore_openAI = init_db()

st.divider()


with open("faiss_store_openai.pkl", "wb") as f:
    pickle.dump(vectorStore_openAI, f)
with open("faiss_store_openai.pkl", "rb") as f:
    VectorStore = pickle.load(f)
llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=VectorStore.as_retriever())

prompt = st.text_input('Ask your question here') 

if prompt:
    with st.spinner('Querying the language model and searching docs. This will take a bit...'):
        outputs = chain({"question": prompt}, return_only_outputs=True)

        st.write("Answer: ", outputs['answer'])

        try:
            sources = outputs['sources']
        except:
            sources = outputs['source']

        if len(sources) > 0 and sources != None:
            st.write("Sources: ", outputs['sources'])

    st.success('Done!')


