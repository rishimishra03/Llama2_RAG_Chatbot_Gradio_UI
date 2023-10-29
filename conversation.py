
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


def create_conversation() -> ConversationalRetrievalChain:

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-roberta-large-v1")

    vectordb = FAISS.load_local("./db/faiss_index", embedding_model)

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=False
    )

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path="./llama-2-7b-chat.ggmlv3.q4_1.bin",
        temperature=0.1,
        top_p=1,
        top_k=250,
        # n_threads=8,
        n_ctx=2048,
        max_tokens=2048,
        repeat_penalty=1.1,
        callback_manager=callback_manager,
        verbose=True,
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=vectordb.as_retriever(),
        memory=memory,
        get_chat_history=lambda h: h,
        verbose=True
    )
    return qa
