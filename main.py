from dotenv import load_dotenv
load_dotenv()
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore=Chroma(persist_directory="chroma_db", embedding_function=embedding_model)


retriever=vectorstore.as_retriever(search_type="mmr",search_kwargs={"k":4,
                                                                    "fetch_k":10,
                                                                    "lambda_mult":0.5,})


llm=ChatGroq(model="openai/gpt-oss-120b")
prompt=ChatPromptTemplate.from_messages([
    ("system", "You are an AI that summarizes the text. Use only the information provided in the text to answer the question. If the answer is not in the text, say 'I don't know'."),
    ("human", """
     Context: {context}
     
     Question: {question}"""),
])
print("Rag system created")
print("Enter 0 to exit")
while True:
    query=input("You:")
    if query=="0":
        break
    docs=retriever.invoke(query)
    context="\n\n".join(
        [doc.page_content for doc  in docs]
    )
    final_prompt=prompt.invoke({
        "context":context,
        "question":query
    })
    response=llm.invoke(final_prompt)
    print(f"\n AI: {response.content}\n")
