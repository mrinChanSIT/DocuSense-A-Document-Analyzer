# s3 bucket name - docusense-a-pdf-analyzer

import boto3
import streamlit as st
import os
import uuid

# s3_client
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

#Bedrock
from langchain_community.embeddings import BedrockEmbeddings

#textsplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

#PDF loader
from langchain.document_loaders import PyPDFLoader

# Importing Vector store
from langchain_community.vectorstores import FAISS

# Creating the instances of Bedrock for embeddings
bedrock_client = boto3.client(service_name = "bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock_client)

# get uuid
def get_uuid():
    return str(uuid.uuid4())

#split the pages into chunks
def chunkify_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

# Create vector store
def create_vector_store(request_id, split_documents):
    faiss_vectorstore = FAISS.from_documents(split_documents, bedrock_embeddings)   # creating the vector embedding store
    file_name = f"{request_id}.bin"
    folder_path = "/tmp/"
    faiss_vectorstore.save_local(index_name=file_name, folder_path=folder_path)

    ## upload to S3
    s3_client.upload_file(Filename=folder_path+'/'+file_name + '.faiss',Bucket=BUCKET_NAME, Key = "my_faiss.faiss")
    s3_client.upload_file(Filename=folder_path+'/'+file_name + '.pkl',Bucket=BUCKET_NAME, Key = "my_faiss.pkl")

    return True


#  main method
def main():
    st.write("This is the admin app test")
    uploaded_file = st.file_uploader("Choose a file", "pdf")
    if uploaded_file is not None:
        req_id = get_uuid()
        st.write(f"Req_id: {req_id}")
        saved_file = f"{req_id}.pdf"
        with open(saved_file, mode="wb") as w:
            w.write(uploaded_file.getvalue())
        
        loader = PyPDFLoader(saved_file)
        pages = loader.load_and_split()

        st.write(f"Total Pages: {len(pages)}")
       

        #split text
        chunkified_docs = chunkify_text(pages,1000,200)

        # create vector store
        st.write("creating the Vector embedding store")
        isVectorized = create_vector_store(req_id,chunkified_docs)

        if isVectorized:
            st.write("Processing document Successful !!!...")
        else:
            st.write("...!!! ERROR !!!... Please recheck")



if __name__ == "__main__":
    main()
