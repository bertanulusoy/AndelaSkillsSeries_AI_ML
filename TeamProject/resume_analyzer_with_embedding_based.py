"""
The below Streamlit application implements an embedding-based search system
"""

import os
import json

import pandas
import streamlit as st
import ast
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens in this program
from scipy import spatial  # for calculating vector similarities for search

import openai


EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"


def load_resume(file_path: str):
    """
    This function is used to load the resume from the file path
    :param file_path: The file path of the resume
    :return: Returns the resume data
    """
    """
    This function is used to load the document from the file path
    :param file_path: The file path of the document
    :return: Returns the document data
    """
    loader = None
    name, extension = os.path.splitext(file_path)
    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file_path}')
        loader = PyPDFLoader(file_path)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file_path}')
        loader = Docx2txtLoader(file_path)
    elif extension == '.txt':
        from langchain_community.document_loaders import TextLoader
        print(f'Loading {file_path}')
        loader = TextLoader(file_path)
    else:
        print(f'The file extension of {extension} does not supported')
    return loader.load()


def chunk_data(data, chunk_size=256, chunk_overlap=20):
    """
    This function is used to chunk the data into smaller pieces
    :param data: The data to be chunked
    :param chunk_size: The size of the chunk
    :param chunk_overlap: The overlap between the chunks
    :return:
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(data)


def clear_history():
    if 'history' in st.session_state:
        del st.session_state.history


def enable_components():
    st.session_state["disabled"] = False


def clear_questions_answers():
    if 'questions' in st.session_state:
        st.session_state.questions = []
    if 'answers' in st.session_state:
        st.session_state.answers = []


def append_question():
    if 'input_question' in st.session_state:
        # question = {'role': 'user', 'content': st.session_state.input_question}
        st.session_state.questions.append(st.session_state.input_question)
        st.session_state.input_question = ""


def prepare_the_embeddings_of_data(chunked_data, model="text-embedding-3-small"):
    texts = [doc.page_content for doc in chunked_data]  # Extract text from each chunk
    embeddings = []

    for text in texts:
        # text = text.replace("\n", " ")  # Prepare text for embedding
        response = openai.embeddings.create(input=[text], model=model)
        embeddings.append(response.data[0].embedding)

    df = pd.DataFrame({
        'text': texts,
        'embedding': embeddings
    })
    # df['embedding'] = df['embedding'].apply(ast.literal_eval)
    return df
    # df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-3-small'))


# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    from openai import OpenAI
    client = OpenAI()
    query_embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding

    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, _ = strings_ranked_by_relatedness(query, df)
    introduction = ('Use the below resume to answer the subsequent question. '
                    'Prepare your answers based on the questions asked previously.'
                    'If the answer cannot be found in the resume, '
                    'write "I could not find an answer."')
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_resume = f'\n\nResume section:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_resume + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_resume
    return message + question


def ask(
    query: str,
    df: pd.DataFrame,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about the resume."},
        {"role": "user", "content": message},
    ]

    for question, answer in zip(st.session_state.questions, st.session_state.answers):
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})

    print(messages)

    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    # print(response.choices[0].message.content)
    return response.choices[0].message.content


def answer_with_ai():
    append_question()
    # data_frame: pandas.DataFrame = st.session_state.df
    for question in st.session_state.questions:
        response = ask(question, st.session_state.df)
        # print(f"Response {response}")
        st.session_state.answers.append(response)
        # print(f"Answers: {st.session_state.answers}")


if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv(), override=True)

    if 'questions' not in st.session_state:
        st.session_state.questions = []
    if 'answers' not in st.session_state:
        st.session_state.answers = []
    if 'df' not in st.session_state:
        st.session_state.df = pandas.DataFrame
    if "disabled" not in st.session_state:
        st.session_state.disabled = True

    st.image('files/img.png')
    st.subheader('Resume Analyzer with embedding-based search system')
    with st.sidebar:
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
        uploaded_file = st.file_uploader('Upload your resume:', type=['pdf', 'docx', 'txt'])
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)
        add_data = st.button('Load resume data', on_click=clear_history)

        if uploaded_file and add_data:
            with st.spinner('Reading, chunking and embedding the resume...'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./files/', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)
                data = load_resume(file_path=file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.session_state.df = prepare_the_embeddings_of_data(chunked_data=chunks, model="text-embedding-ada-002")
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')
                if data:
                    enable_components()

    st.divider()
    st.text_input('Ask question(s), click `Analyse` button to begin the analysis:',
                  on_change=answer_with_ai,
                  disabled=st.session_state.disabled,
                  key='input_question')

    # to display questions and answers matched side by side
    st.write("Q&A Matched:")
    for question, answer in zip(st.session_state.questions, st.session_state.answers):
        # Columns for questions and answers
        question_col, answer_col = st.columns(2)
        with question_col:
            st.write(question)
        with answer_col:
            st.write(answer)
    # clear_questions_answers()
