from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
import os

# 1. Configuração inicial
print("🔍 Iniciando sistema RAG...")

# 2. Carregar documentos PDF
try:
    loader = PyPDFLoader("docs/heroes_tec.pdf")
    documents = loader.load()
    print(f"✅ PDF carregado - {len(documents)} páginas")
except Exception as e:
    print(f"❌ Erro ao carregar PDF: {str(e)}")
    print("Verifique se:")
    print("- O arquivo existe em 'docs/heroes_tec.pdf'")
    print("- O PDF não está protegido por senha")
    exit()

# 3. Pré-processamento otimizado para tabelas
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n", "|", "  ", "SKU"]  # Separa por linhas de tabela
)
texts = text_splitter.split_documents(documents)
print(f"✂️ Documento dividido em {len(texts)} trechos")

# 4. Configuração de embeddings
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
except Exception as e:
    print(f"❌ Erro nos embeddings: {str(e)}")
    print("Tente: pip install --upgrade sentence-transformers")
    exit()

# 5. Banco de dados vetorial
db = FAISS.from_documents(texts, embeddings)

# 6. Conexão com o Ollama
try:
    llm = Ollama(
        model="llama3",
        base_url="http://localhost:11434",
        temperature=0.3  # Reduz criatividade para respostas precisas
    )
    print("🦙 Modelo Llama3 conectado")
except Exception as e:
    print(f"❌ Erro no Ollama: {str(e)}")
    print("Verifique se:")
    print("- Ollama está rodando (execute 'ollama serve' em outro terminal)")
    print("- O modelo está baixado (execute 'ollama pull llama3')")
    exit()

# 7. Template de prompt para tabelas
template = """Você é um assistente técnico. Responda APENAS com os dados exatos desta tabela:

{context}

Pergunta: {question}

Resposta direta da tabela:"""
prompt = ChatPromptTemplate.from_template(template)

# 8. Cadeia RAG
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# 9. Interface de consulta
print("\n💡 Sistema pronto! (Digite 'sair' para encerrar)")
while True:
    try:
        pergunta = input("\nPergunta sobre SKUs: ")
        if pergunta.lower() == 'sair':
            break
            
        resposta = qa.invoke({"query": pergunta})
        
        # Exibe resposta formatada
        print(f"\n🔍 Resposta: {resposta['result']}")
        print(f"📌 Fonte: {resposta['source_documents'][0].metadata['source']} (página {resposta['source_documents'][0].metadata.get('page', 'N/A')})")
        
    except Exception as e:
        print(f"⚠️ Erro: {str(e)}")