from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

'''
Loading documents from /data directory mainly the clinical sop pdf's
'''


def structure_documents():
    dir_loader = DirectoryLoader(
        "./data",
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=False,
    )

    docs = dir_loader.load()
    print(f"Loaded {len(docs)} pages")
    return docs




