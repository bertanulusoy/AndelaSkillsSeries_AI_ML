import streamlit as st
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
import json

# Setup for the chat
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=1)
history = FileChatMessageHistory('chat_history.json')
memory = ConversationBufferMemory(
    memory_key='chat_history',
    chat_memory=history,
    return_messages=True
)
prompt = ChatPromptTemplate(
    input_variables=['content'],
    messages=[
        SystemMessage(content='You are a chatbot having a conversation with a human.'),
        MessagesPlaceholder(variable_name='chat_history'),
        HumanMessagePromptTemplate.from_template('{content}')
    ]
)
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=False
)

# Streamlit app
st.set_page_config(layout="wide")


def read_conversation_history():
    try:
        with open('chat_history.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []


def display_conversation(conversation_data):
    for entry in conversation_data:
        # Displaying each entry in the main column
        st.markdown(f"**{entry['type'].capitalize()}**: {entry['data']['content']}")


# Display conversation history
conversation_data = read_conversation_history()
display_conversation(conversation_data)


# Place the prompt input and send button at the bottom
with st.container():
    user_input = st.text_input("Your Prompt", key="user_input")  # Using a key to clear after sending
    if st.button("Send"):
        response = chain.run({'content': user_input})
        conversation_data = read_conversation_history()  # Reload the updated conversation
        # Clear the existing elements and redisplay updated conversation at the top
        st.experimental_rerun()

