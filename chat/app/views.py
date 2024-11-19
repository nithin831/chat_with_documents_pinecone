from django.shortcuts import render, redirect
from django.http import JsonResponse
import os
from dotenv import load_dotenv
load_dotenv()
from django.conf import settings
import tempfile



from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
import pinecone 
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain

from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter



import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.data.path.append('/tmp/nltk_data')

# Create your views here.
MISTRAL_API_KEY = os.environ["app"]

def doc(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('file')
        file_type = request.POST.get('fileType')
        file_name = request.POST.get('fileName')
        file_extension = request.POST.get('fileExtension')
        file_size_no = uploaded_file.size/ 1024
        file_size = f"{round(file_size_no, 2)} KB"
        print(file_size)
        if not uploaded_file:
            return JsonResponse({'error': 'Please upload a file'}, status=400)
            
         # Create a temporary file to hold the uploaded content
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        
        # Determine file type and load accordingly
        if uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(temp_file_path)
        elif uploaded_file.name.endswith('.txt'):
            loader = TextLoader(temp_file_path)
        elif uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
            loader = UnstructuredExcelLoader(temp_file_path, mode="elements")
        elif uploaded_file.name.endswith('.csv'):
            loader = UnstructuredCSVLoader(temp_file_path, mode="elements")
        elif uploaded_file.name.endswith('.docx'):
            loader = UnstructuredWordDocumentLoader(temp_file_path, mode="elements")
        elif uploaded_file.name.endswith('.md'):
            loader = UnstructuredMarkdownLoader(temp_file_path, mode="elements")

        else:
            os.remove(temp_file_path)  # Clean up the temporary file
            return JsonResponse({'error': 'Unsupported file type'}, status=400) 
        
        docs = loader.load_and_split()

        # Split text into chunks 
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        print("documents", len(documents))
        # Define the embedding model
        embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=MISTRAL_API_KEY)
        embedding_example = embeddings.embed_documents([documents[0].page_content])
        print(len(embedding_example[0])) 
        pinecone.init(
            api_key=os.environ["PINECONE_API_KEY"],
            environment=os.environ["PINECONE_ENVIRONMENT"]
        )
        index_name = 'chat-with-files'
        # index_info = pinecone.describe_index(index_name)

        # # Print the dimension of the index
        # print(f"Pinecone index dimension: {index_info}")
        # Check if index exists and delete it if present
        if index_name in pinecone.list_indexes():
            pinecone.delete_index(index_name)
            print(f"Deleted existing index: {index_name}")

        # Create a new index
        pinecone.create_index(
            index_name,
            dimension=1024,
            metric='cosine'
        )

        # connect to the index
        
        global index
        index=Pinecone.from_documents(documents,embeddings,index_name=index_name)
        
        os.remove(temp_file_path)
        # Pass file metadata to the context
        global context
        context = {
            'file_name': file_name,
            'file_size': file_size,
            'file_type': file_type,
            'file_extension': file_extension,
        }  
        return redirect('chatbot')

    return render(request, 'doc.html')
    

def ask_mistralai(message):
    
    # Define LLM
    model = ChatMistralAI(model="mistral-large-latest", mistral_api_key=MISTRAL_API_KEY)
    matching_results=index.similarity_search(message,k=2)
    chain=load_qa_chain(model,chain_type="stuff")
    response=chain.run(input_documents=matching_results,question=message)
    print(response)
    return response
   

def chatbot(request):

    if request.method == 'POST':
        message = request.POST.get('message')
        response = ask_mistralai(message)

        return JsonResponse({'message': message, 'response': response})
    return render(request, 'chat.html',context)
