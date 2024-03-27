from langchain_openai import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
import streamlit as st
from streamlit_chat import message

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

st.set_page_config(
    page_title='Your Custom Assistant',
    page_icon='🤖'
)
st.subheader('Your Custom ChatGPT 🤖')
chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5)
if 'message' not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    system_message = st.text_input(label='System role')
    user_prompt = st.text_input(label='Send a message')
    if system_message:
        # If session_state.messages has not a message type of SystemMessage,
        # the new system_message will be appended to the session_state.messages list.
        if not any(isinstance(x, SystemMessage) for x in st.session_state.messages):
            st.session_state.messages.append(
                SystemMessage(content=system_message)
            )
        # st.write(st.session_state.messages)

    if user_prompt:
        st.session_state.messages.append(
            HumanMessage(content=user_prompt)
        )

        with st.spinner('Working on you request ... '):
            response = chat.invoke(st.session_state.messages)

        st.session_state.messages.append(AIMessage(content=response.content))

# st.session_state.messages
# message('This is chatgpt', is_user=False)
# message('This is the user', is_user=True)

if len(st.session_state.messages) >= 1:
    if not isinstance(st.session_state.messages[0], SystemMessage):
        st.session_state.messages.insert(
            0,
            SystemMessage(
                content='You are a helpful assistant.')
        )

# The for loop is starting from the first item of the session_state.messages because,
# the st.session_state.messages[0] is the SystemMessage. We don't want to display it.
for i, msg in enumerate(st.session_state.messages[1:]):
    if i % 2 == 0:
        message(msg.content, is_user=True, key=f'{i} +  🤓')
    else:
        message(msg.content, is_user=False, key=f'{i} +   🤖')

