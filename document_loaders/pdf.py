from langchain_community.document_loaders import PyPDFLoader
data=PyPDFLoader("document_loaders/NIPS-2017-attention-is-all-you-need-Paper.pdf")
docs=data.load()
print(len(docs))