from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
import os

# 1. Configuração especial para PDFs tabelados
loaders = {
    '.pdf': PyPDFLoader,
    '.txt': TextLoader
}

loader = DirectoryLoader(
    "./docs",
    loader_kwargs={"extract_images": False},  # Otimiza para tabelas
    use_multithreading=True,
    loaders=loaders,
    silent_errors=True
)

# 2. Processamento otimizado para tabelas
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", "|", "  "]  # Separa por células de tabela
)

# 3. Carrega e divide documentos
documents = loader.load()
texts = text_splitter.split_documents(documents)

# 4. Embeddings com foco em dados estruturados
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

# 5. Banco vetorial
db = FAISS.from_documents(texts, embeddings)

# 6. Template especial para tabelas
TEMPLATE = """Você é um especialista em análise de tabelas. Responda com EXATAMENTE o conteúdo da tabela quando disponível.

Contexto:
{context}

Pergunta: {question}

Resposta baseada estritamente na tabela:"""
prompt = ChatPromptTemplate.from_template(TEMPLATE)

# 7. Cadeia RAG
qa = RetrievalQA.from_chain_type(
    llm=Ollama(model="llama3", temperature=0),
    retriever=db.as_retriever(search_kwargs={"k": 3, "score_threshold": 0.4}),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# Interface
print("Sistema RAG para Tabelas - Pronto")
while True:
    question = input("\nPergunta sobre SKUs (ou 'sair'): ")
    if question.lower() == 'sair':
        break
    
    result = qa.invoke({"query": question})
    
    if "SKU" in result["result"]:  # Resposta contém dados tabulares
        print(f"\n✅ Resposta Técnica:")
        print(result["result"])
    else:
        print(f"\nℹ️ Resposta Geral:")
        print(result["result"])
    
    print(f"\n🔍 Fonte: {result['source_documents'][0].metadata['source']}")