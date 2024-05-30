# s3 bucket name - docusense-a-pdf-analyzer

import boto3
import streamlit as st
import os
import uuid

# s3_client
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

#Bedrock
from langchain.llms.bedrock import Bedrock
from langchain_community.embeddings import BedrockEmbeddings

# Prompt templates and chains
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Importing Vector store
from langchain_community.vectorstores import FAISS

# Creating the instances of Bedrock for embeddings
bedrock_client = boto3.client(service_name = "bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock_client)
folder_path = "/tmp/"
# get uuid
def get_uuid():
    return str(uuid.uuid4())

#split the pages into chunks

# Create vector store


def load_index():
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss", Filename=f"{folder_path}my_faiss.faiss")
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl", Filename=f"{folder_path}my_faiss.pkl")

def get_llm():
    llm = Bedrock(model_id="anthropic.claude-v2:1", client=bedrock_client, model_kwargs={'max_tokens_to_sample':512})

    return llm

def get_response(llm, vectorstore,question):
    ## create prompt / template
    prompt_template = """

    Human: Please use the given context to provide concise answer to the question
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>

    Question: {question}

    Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":question})
    return answer['result']





#  main method
def main():
    st.header("This is the user app test using Bedrock,RAG..")
    # load index
    load_index()
    
    dir_list = os.listdir(folder_path)
    st.write(f"File and directory in the {folder_path}")
    st.write(dir_list)

    # Create Index
    faiss_index = FAISS.load_local( 
        index_name="my_faiss", 
        folder_path=folder_path,
        embeddings=bedrock_embeddings,
        allow_dangerous_deserialization=True)
    
    st.write(f"!!!... INDEX IS READY ...!!!")
    question = st.text_input("Ask your Question...?")
    if st.button(f"Ask Question..."):
        with st.spinner(f"Querying...."):

            # get the llm
            llm = get_llm()

            # get_response
            st.write(get_response(llm,faiss_index,question))
            st.success(f"Here is the response for your question based on my understanding of the document provided.")
            

if __name__ == "__main__":
    main()
