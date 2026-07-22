from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
import json

load_dotenv()

llm = ChatGroq(model='llama-3.3-70b-versatile')

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory='RAG_vectorstore_test_db',
    collection_name='research_papers'
)

parser = StrOutputParser()

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={'k':10}, search_type='similarity'),
    llm = llm
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a helpful AI Research Assistant.

Answer the user's question ONLY using the retrieved context below.

If the retrieved context does not contain sufficient information, respond that you have inadequate information to answer the question.

Retrieved Context:
{context}
"""
    ),

    MessagesPlaceholder(variable_name="chat_history"),

    (
        "human",
        "{question}"
    )
])

def load_history(_):

    history = []
    # load the json file containing the chat history
    with open('chat_history.json', 'r') as f:
        data = json.load(f)

    # append langchain messages inside 'history' list
    for messages in data:
        if messages['role'] == 'user':
            history.append(HumanMessage(content=messages['content']))

        elif messages['role'] == 'ai':
            history.append(AIMessage(content=messages['content']))

    return history

def save_history(user, ai):

    # load the chat history to append it 
    with open('chat_history.json', 'r') as f:
        data = json.load(f)

    # create dictionaries to store new question reply convo
    user_data = {'role': 'user', 'content': user}
    ai_data = {'role': 'ai', 'content': ai}

    # append the data to be rewriten in the chat history
    data.append(user_data)
    data.append(ai_data)

    # redump new data in chat history:
    with open('chat_history.json', 'w') as f:
        json.dump(data, f, indent=4)

# a function to extract only page content of every relevant document object chunk obtained from semantic searching is required that will be passed on to a runnable lambda
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough(),
    'chat_history': RunnableLambda(load_history)
})

chain = parallel_chain | prompt | llm | parser

def get_response(user_input):

    # invoking entire chain of runnables to feed the prompt with contextual learning to the LLM and returning the output and input in save history to save the chat history
    result = chain.invoke(user_input)
    save_history(user_input, result)
    return result







