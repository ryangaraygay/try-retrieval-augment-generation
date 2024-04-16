from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.indexes import VectorstoreIndexCreator
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory

import tiktoken

urls = [
    "https://ryangaraygay.com/about/",
]

loader = UnstructuredURLLoader(urls=urls)
data = loader.load()
data

import os

# Prompt the user for their OpenAI API key
api_key = input("Please enter your OpenAI API key: ")

# Set the API key as an environment variable
os.environ["OPENAI_API_KEY"] = api_key

# Optionally, check that the environment variable was set correctly
print("OPENAI_API_KEY has been set!")

llm_model = "gpt-3.5-turbo"

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
data = text_splitter.split_documents(data)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(data, embedding=embeddings)

# Create conversation chain
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4")
memory = ConversationBufferMemory(
memory_key='chat_history', return_messages=True)
conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        memory=memory
        )

query = input("Please enter your query: ")
result = conversation_chain({"question": query})
answer = result["answer"]
print(answer)