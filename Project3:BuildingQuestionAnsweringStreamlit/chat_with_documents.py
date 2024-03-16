import os
import streamlit as st
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def load_document(file_path):
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


def chunk_data(data, chunk_size=256, chunk_overlap=20) -> list:
    """
    This function is used to chunk the data into smaller pieces
    :param data: The data to be chunked
    :param chunk_size: The size of the chunk
    :param chunk_overlap: The overlap between the chunks
    :return:
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(data)


def create_embeddings(chunks):
    """
    This function is used to create the embeddings for the chunks
    and return the chroma vector store
    :param chunks: The chunks of the document
    :return: Returns the chroma vector store
    """
    embeddings = OpenAIEmbeddings()
    return Chroma.from_documents(chunks, embeddings)


def ask_and_get_answer(vector_store, q, k=3):
    """
    This function is used to ask a question and get the answer
    :param vector_store: The vector store
    :param q: The question to be asked
    :param k: It is the top k most similar chunks of data to be returned
    :return: Returns the answer to the question
    """
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    return chain.invoke(q)


def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')
    return total_tokens, total_tokens / 1000 * 0.0004


def clear_history():
    if 'history' in st.session_state:
        del st.session_state.history


if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    st.image('files/img.png')
    st.subheader('LLM Question-Answering Application')
    with st.sidebar:
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)
        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data:
            with st.    spinner('Reading, chunking and embedding file ...'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./files/', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')

                vector_store = create_embeddings(chunks)

                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully.')

    q = st.text_input('Ask a question about the content of your file:')
    if q:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            # The most similar chunks LLM will use.
            # Higher k leads to an accurate result, but it may take longer to answer
            st.write(f'k: {k}')
            answer = ask_and_get_answer(vector_store, q, k)
            st.text_area('LLM Answer: ', value=answer['result'])
            st.divider()
            if 'history' not in st.session_state:
                st.session_state.history = ''
            value = f'Q: {q} \nA: {answer["result"]}'
            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history
            st.text_area(label='Chat History', value=h, key='history', height=400)
