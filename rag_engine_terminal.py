from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load documents
txt_folder_path = '/Users/andriirak/Documents/Marbet_Challenge/txt_cleaned'
loader = DirectoryLoader(txt_folder_path, glob='*.txt', loader_cls=TextLoader)
documents = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=['---', "\n\n"],
    length_function=len,
    is_separator_regex=False
)
chunks = splitter.split_documents(documents)

# 3. Initialize models
model = ChatOllama(base_url='http://194.171.191.226:3061', model='llama3.3:70b-instruct-q5_K_M')
embedding_model = OllamaEmbeddings(model='mxbai-embed-large')
vectorstore = FAISS.from_documents(chunks, embedding=embedding_model)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 10})

def greet_user():
    print('='*70)
    print('Hello! I am Marbet Bot, your helpful assistant for the incentive trip. \n'
          'How can I assist you today?')
    print('='*70, '\n\n')

# 4. Prompt and chain
template = '''
You are an AI travel assistant for an exclusive incentive trip, specialized in providing detailed and accurate 
information based only on the selected event document.

Rules:
- If you cannot find the answer directly in the provided text chunks, politely say you don't know and encourage the user 
to rephrase or ask differently.
- If the user's question is too broad (e.g., "Tell me everything about the trip"), provide a helpful summary of the most
relevant points you can find, but stay within the retrieved information.
- Never invent, guess, or add information that is not explicitly stated in the document.
- When answering, be clear, detailed, and as precise as possible, helping the user fully understand the event 
activities, logistics, or requirements.
- Prefer structured, organized answers (lists, steps, or short paragraphs) when possible to make the information easier 
to follow.

You must rely solely on the following retrieved information:
{documents}

User's question:
{question}
'''
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

greet_user()
while True:
    print('-'*70)
    question = input('Ask your question (q to quit): ')
    print('-'*70,'\n\n')
    if question == 'q':
        break

    relevant_chunks = retriever.invoke(question)
    flattened_chunks = '\n\n'.join(doc.page_content for doc in relevant_chunks)
    result = chain.invoke({
        'documents': flattened_chunks,
        'question': question
    })
    print(result.content)
