from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents:list, chunk_size = 1000, chunk_overlap = 200):


    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
    )

    split_documents = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_documents)} chunks")
    return split_documents

    