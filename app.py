import streamlit as st
from pypdf import PdfReader
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import json
import time
import requests
from PIL import Image
import  google.generativeai as genai
from io import BytesIO
from youtube_transcript_api import YouTubeTranscriptApi

#Gemini API configration
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

#Function to get the PDF dcouments
def get_pdf_text(pdf_docs):
    text=""
    pdf_reader=PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text+=page.extract_text()
    return text

#Function to extract the text from provided PDF
def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks

#Function to embedding the text and storing the text chunks
def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local('faiss_index')

#Create the prompt and import the model
def get_conversational_chain():
    prompt_template= """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "Modify your question to find the related content in the document", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question:\n {question}\n

    Answer: 
    """
    model=ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.3)
    promt=PromptTemplate(template=prompt_template, input_variables=['context','question'])
    chain=load_qa_chain(model,chain_type="stuff", prompt=promt)
    return chain

#Function to take the user input and generate the response
def user_input(user_question):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db=FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)

    docs=new_db.similarity_search(user_question)
    
    chain= get_conversational_chain()
    
    response=chain(
        {"input_documents":docs,"question":user_question}, return_only_outputs=True)
    st.session_state.response = response['output_text']

#Import the model to genrate the response text based search
model=genai.GenerativeModel("gemini-2.0-flash")
chat=model.start_chat(history=[])

#Function to genrate the response for  user text
def get_gimini_response_txt(question):
    response=chat.send_message(question)
    return response

#Import the model to genrate the response image based Q&A
vision_model= genai.GenerativeModel("gemini-2.0-flash")

#Function to genrate the response for  user text
def get_gimini_response(user_text, image,prompt):
        response = vision_model.generate_content([user_text,image,prompt], stream=True)
        return response

#prompt  for youtube video summary
prompt="""You are an expert youtube summarizer.Produce a detailed summary of the YouTube video titled '[Insert Video Title].'
 Aim for a comprehensive summary with final conclusion """

#Function to extract the texts from youtube
def extract_transcript_details(youtube_video_url):
    try:
        video_id=youtube_video_url.split("=")[1]
        transcript_text=YouTubeTranscriptApi.get_transcript(video_id)
        transcript=""
        for i in transcript_text:
            transcript += i['text']
        return transcript
    except Exception as e:
        raise e

#Function to genrate the youtube summary   
def genrate_yt_content(transcript_text,prompt):
    model=genai.GenerativeModel("gemini-2.0-flash")
    response=model.generate_content(prompt+transcript_text)
    return response.text

#Function to create the Q&A model for youtube video    
def get_text_and_conversational_chain():
    prompt_template= """
    You are an expert youtube video summarizer and your role is to answer the question as detailed as possible from the provided context, 
    make sure to provide all the details, if the answer is not in provided context just say, 
    "Modify your question to find the related content in the video", don't provide the wrong answer,you will get $100 as tip if you provided good answer\n\n
    Context:\n {context}?\n
    Question:\n {question}\n

    Answer: 
    """
    model=ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.3)
    promt=PromptTemplate(template=prompt_template, input_variables=['context','question'])
    chain=load_qa_chain(model,chain_type="stuff", prompt=promt)
    return chain

#Function to take the user input and genrate the response
def user_query(user_question):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db=FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)

    docs=new_db.similarity_search(user_question)
    
    chain= get_text_and_conversational_chain()
    
    response=chain(
        {"input_documents":docs,"question":user_question}, return_only_outputs=True)
    st.session_state.response = response['output_text']

#Function to stream the animations
def load_lottiefiles(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

#Set the page config
st.set_page_config("PrivAI ",layout="wide",page_icon=":vhs:")


lottie_hi = load_lottiefiles(r'Images/AI.json')
st_lottie(
        lottie_hi, loop=True, quality="high", speed=1.65, key=None, height=100)

#Home page functions     
def home_page():

    st.markdown("""# <span style='color:#0A2647'>Welcome to  PrivAI  </span>""", unsafe_allow_html=True)
    st.divider()
    col1,col2=st.columns([0.6,0.4],gap="large")
    with col1:
        
        st.markdown("""## <span style='color:#AE194B'>Introduction</span>""", unsafe_allow_html=True)

        st.markdown("""
            ###          
            <span style='color:#0D7177'>  **PrivAI**, being **Google-powered** and based on the **Gemini AI** architecture, 
            represents a state-of-the-art large language model capable of understanding and generating human-like 
            text for a wide range of natural language processing tasks,A large language model,
            Google Gemini is a family of AI models that can understand and work with text, images, audio, videos, and code.
            Gemini LLM harnesses the power of advanced natural language processing to provide you with a personalized and 
            efficient browsing experience. With its state-of-the-art capabilities, Gemini LLM brings forth four distinct services, 
            each tailored to cater to your diverse needs.
                    
            <br>

            Experience the next generation of online interaction with Gemini LLM and its array of services. 
            Whether you're seeking information, analyzing images, exploring documents, 
            or delving into videos, we've got you covered. Get ready to embark on a journey of discovery like never before!</span>""", 
            unsafe_allow_html=True)
           
    

    # st.divider()
    # st.markdown("""## <span style='color:#9D801C'>Services Usage Demo</span>""", unsafe_allow_html=True)
    # st.divider()
    # Text_chat,Image_chat,PDF_chat,Video_chat=st.columns(4,gap="large")
    # with Text_chat:
    #     st.markdown("""
    #         ###          
    #         <span style='color:#0D7177'>  📝 **Navigate to TEXT CHAT** : Here user input their message to get the response this will genrate the Text based results, 
    #                     The text-based responses provided by our AI system are intended for informational purposes only. They are not a substitute for professional advice, 
    #                     consultation, or services. If you require specific assistance or guidance in areas such as medical, legal, financial, or other specialized domains, 
    #                     we recommend consulting qualified professionals.</span>""", unsafe_allow_html=True)
    #     st.divider()
    #     st.markdown("**Please watch the Demo**")
    #     st.video(r"Videos/Textchat.mp4")
       
    # with Image_chat:
    #     st.markdown("""
    #         ###                            
    #         <span style='color:#34786A'>  🛣️ **Navigate to IMAGE CHAT**: Here user will provide the images and ask the AI system to query,IMAGE based results 
    #                     The Gemini model integrates the information extracted from images with language understanding to provide contextually relevant 
    #                     and informative responses.While our AI system aims to provide informative and engaging responses, 
    #                     users should verify the information provided, especially in critical or sensitive situations.User can extract the content in the image
    #                 </span>""", unsafe_allow_html=True)
    #     st.divider()
    #     st.markdown("**Please watch the Demo**")
    #     st.video(r"Videos/Imagechat.mp4")

    # with PDF_chat:
    #     st.markdown("""
    #         ###           
    #         <span style='color:#365B4F'>  📚 **Navigate PDF CHAT**: Here user will upload the pdf and submit to process the pdf to convert into text,
    #                     and ask your question the AI system will genrate the answer based on avilable context,
    #                     Our Q&A model is designed to analyze PDF documents and extract valuable information to answer your questions accurately. 
    #                     The Gemini model leverages the content within PDFs to generate contextually relevant responses to your inquiries</span>""", unsafe_allow_html=True)
    #     st.divider()
    #     st.markdown("**Please watch the Demo**")
    #     st.video(r"Videos/PDFchat.mp4")
        

    # with Video_chat:
    #     st.markdown("""
    #         ###                       
    #         <span style='color:#2A5D87'>  📽️ **Navigate to VIDEO CHAT**: Here user will provide the youtube video link to extract the content from 
    #                     video and ask the question about the video, our AI system will genrate the answer based on the available context.
    #                     Our Q&A model is designed to analyze video content and extract valuable information to answer your questions accurately. 
    #                     The Gimini model leverages the visual and auditory context within videos to generate contextually relevant responses
    #                     to your inquiries.  </span>
    #         """, unsafe_allow_html=True)
    #     st.divider()
    #     st.markdown("**Please watch the Demo**")
    #     st.video(r"Videos/Videochat.mp4")



# TESTING PROTTYPE FOR PDF CHAT
def build_chat_history_prompt(chat_history, current_question):
    prompt = ""
    for msg in chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        prompt += f"{role}: {msg['content']}\n"
    prompt += f"User: {current_question}\nAssistant:"
    return prompt

def chat_with_multipdf():
    # --- Session State Initialization ---
    if "chat_history_pdf" not in st.session_state:
        st.session_state.chat_history_pdf = []
    if "response" not in st.session_state:
        st.session_state.response = ""
    if "pdf_text" not in st.session_state:
        st.session_state.pdf_text = ""

    st.header("Multi-PDF's 📚 - Chat Agent 🤖")
    st.markdown(
        "<span style='color:#EE8E8E'>(Once you Click on the Home button 🏠 your chat history will be <b>Deleted</b>)</span>",
        unsafe_allow_html=True
    )

    user_question = st.chat_input(placeholder="Ask a Question from the PDF Files uploaded .. ✍️📝")

    # --- Initial Prompt ---
    if not st.session_state.chat_history_pdf:
        with st.chat_message("user"):
            st.markdown("Once the PDF is uploaded, please write your question 👇")

    # --- Generate and Store Response ---
    if user_question:
        with st.spinner('Please wait, generating answer... 📖'):
            try:
                # Try to answer using PDF-based context (RAG)
                user_input(user_question)
                answer = st.session_state.response

                # Fallback to Gemini if RAG fails
                if "Modify your question to find the related content in the video" in answer:
                    # Build chat history prompt for Gemini
                    prompt = build_chat_history_prompt(
                    st.session_state.chat_history_video, user_question
                    )
                    # Optionally, add PDF content for context (if not too large)
                    if st.session_state.transcript_text:
                        prompt += f"\n\nRelevant video content (if needed):\n{st.session_state.transcript_text[:8000]}"
                        answer = get_gimini_response_txt(prompt)
                        if hasattr(answer, "text"):
                            answer = answer.text

                st.session_state.response = answer
                st.session_state.chat_history_pdf.append({"role": "user", "content": user_question})
                st.session_state.chat_history_pdf.append({"role": "assistant", "content": st.session_state.response})
            except Exception as e:
                st.error(f"Error generating response: {e}")

    # --- Display Chat History ---
    for message in st.session_state.chat_history_pdf:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Sidebar: PDF Upload Section ---
    with st.sidebar:
        st.title("📁 PDF File's Section")
        lottie_hi = load_lottiefiles(r'Images/PDF.json')
        st_lottie(lottie_hi, loop=True, quality="high", speed=1.65, height=100)
        st.divider()

        pdf_docs = st.file_uploader(
            "Upload your PDF Files & \nClick on the Submit & Process Button",
            type=["pdf"]
        )

        if pdf_docs is not None:
            if not pdf_docs.name.lower().endswith('.pdf'):
                st.warning("Uploaded file is not in PDF format. Please upload a PDF file.")
            elif st.button("Submit & Process"):
                with st.status("Processing....", expanded=True) as status:
                    try:
                        st.write("Extracting Text...")
                        time.sleep(2)
                        raw_text = get_pdf_text(pdf_docs)
                        st.session_state.pdf_text = raw_text

                        st.write("Converting Text into embeddings...")
                        time.sleep(1)
                        text_chunks = get_text_chunks(raw_text)

                        st.write("Storing all the chunks...")
                        time.sleep(1)
                        get_vector_store(text_chunks)

                        status.update(label="Successfully processed", state="complete", expanded=False)
                        st.success("PDF processed and ready for questions!")
                    except Exception as e:
                        status.update(label="Processing failed", state="error", expanded=True)
                        st.error(f"Error processing PDF: {e}")

# ENDS HERE


#Function to chat with pdf documents
# def chat_with_multipdf():

#     #Initialize the session for chat
#     if "chat_history_pdf" not in st.session_state:
#         st.session_state.chat_history_pdf = []

#     st.header("Multi-PDF's 📚 - Chat Agent 🤖 ")
#     st.markdown("""<span style='color:#EE8E8E'>(Once you Click on the Home button 🏠 your chat history will be **Deleted**)</span>""", unsafe_allow_html=True)

#     #Take the user input
#     user_question=st.chat_input(placeholder="Ask a Question from the PDF Files uploaded .. ✍️📝")

#     with st.chat_message("user"):
#             st.markdown("Once the PDF is  uplodaded please write your question 👇")

#     #Genrate the response 
#     if user_question: 
#         with st.spinner('Please wait Generating answer.....📖'):
#             user_input(user_question)
    
#              #   Append question and answer to chat history
#             st.session_state.chat_history_pdf.append({"role": "user", "content": user_question})
#             st.session_state.chat_history_pdf.append({"role": "assistant", "content": st.session_state.response}) 

#                 # Display chat history
#     for message in st.session_state.chat_history_pdf:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"]) 

#     #Take the PDF documents from the user                 
#     with st.sidebar:
#         st.title("📁 PDF File's Section")
#         lottie_hi = load_lottiefiles(r'Images/PDF.json')
#         st_lottie(
#             lottie_hi, loop=True, quality="high", speed=1.65, key=None, height=100)
#         st.divider()
#         pdf_docs = st.file_uploader("Upload your PDF Files & \n Click on the Submit & Process Button ")

#         if pdf_docs is not None:
#             if not pdf_docs.name.endswith('.pdf'):
#                 st.warning("Uploaded file is not PDF format,Please upload a PDF file.")
#             elif st.button("Submit & Process"):
#                 with st.status("Processing....",expanded=True) as status:
#                     st.write("Extracting Text...")
#                     time.sleep(2)
#                     st.write("Converting Text into emmeddings")
#                     time.sleep(1)
#                     st.write("Storing all the chunks")
#                     time.sleep(1)
#                     status.update(label="Sucessfully processed", state="complete", expanded=False)
#                     raw_text = get_pdf_text(pdf_docs)
#                     text_chunks= get_text_chunks(raw_text)
#                     get_vector_store(text_chunks)  

#TESTING CHAT FUNCTIONALITY 
def build_chat_history_prompt(chat_history, current_question):
    prompt_text = ""
    # Build history from previous turns
    for msg in chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        prompt_text += f"{role}: {msg['content']}\n"
    # Add the current question
    prompt_text += f"User: {current_question}\nAssistant:"
    return prompt_text

def text_chat():
    st.header("Your AI powered Chat Agent 🤖 ")
    st.markdown(
        "<span style='color:#EE8E8E'>(Once you Click on the Home button 🏠 your chat history will be <b>Deleted</b>)</span>",
        unsafe_allow_html=True
    )

    # Initialize the session for chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # --- Display existing chat messages ---
    # This loop runs on every rerun, showing the full history up to that point
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Get User Input ---
    prompt = st.chat_input("Ask a Question your AI chat Agent ✍️")

    if prompt:
        # --- Display User Message Immediately ---
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to session state chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # --- Generate and Display Assistant Response ---
        with st.spinner('Generating response...'):
            try:
                # Build full chat history prompt for Gemini
                # Pass the history *before* the current user message was added
                question = build_chat_history_prompt(st.session_state.chat_history[:-1], prompt)

                # Get Gemini response
                response = get_gimini_response_txt(question)
                if hasattr(response, "text"):
                    full_response = response.text
                else:
                    full_response = response # Assuming it's already a string

                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(full_response)

                # Add assistant response to session state chat history
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})

            except Exception as e:
                error_msg = f"An error occurred during response generation: {str(e)}"
                # Display error message in chat
                with st.chat_message("assistant"):
                    st.error(error_msg)
                # Add error message to session state chat history
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})


#TESTING SECTION ENDS HERE

        
#Function to text based response genration                   
# def text_chat():

#     st.header("Your AI powered Chat Agent 🤖 ")
#     st.markdown("""<span style='color:#EE8E8E'>(Once you Click on the Home button 🏠 your chat history will be **Deleted**)</span>""", unsafe_allow_html=True)

#     with st.chat_message("user"):
#         st.markdown("Hi I am your AI chat Bot Ask me anything 👍")

#     #Initialize the session for chat
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history =[]

#         # Display chat messages from history on app rerun
#     for message in st.session_state.chat_history:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])   

#     if prompt := st.chat_input("Ask a Question your AI chat Agent ✍️"):
#             # Add user message to chat history
#         st.session_state.chat_history.append({"role": "user", "content": prompt})
#             # Append the dialogue history to the user's prompt
#         question = "\n".join([message["content"] for message in st.session_state.chat_history])
#             # Display user message in chat message container
#         with st.chat_message("user"):
#             st.markdown(prompt)
#             # Display assistant response in chat message container
#         with st.spinner('Generating response....'):
#             with st.chat_message("assistant"):
#                 try:
#                     for response in get_gimini_response_txt(question):
#                         full_response=""
#                         for chunk in response:
#                                 full_response += chunk.text + " "
#                                 st.markdown(full_response)
#                                 # Check if there are follow-up questions
#                                 if "?" in prompt:
#                                 # Update the chat history with the assistant's response
#                                     st.session_state.chat_history.append({"role": "assistant", "content": full_response}) 
#                                 # Clear the chat input box
#                                     st.session_state.prompt = ""
#                                 # Set the chat input box value to the assistant's response
#                                     st.chat_input("Follow-up question", value=full_response)
#                             # Update the chat history
#                                 st.session_state.chat_history.append({"role": "assistant", "content": full_response})
#                 except Exception as e:
#                     st.error(f"An error occurred during response generation: {str(e)}")
#                 # Update the chat history with the error message
#                     st.session_state.chat_history.append({"role": "assistant", "content": f"An error occurred: {str(e)}"})

#Functions to genrate the response based on the image
def chat_with_image():
    #Initialize the session for chat
    if "chat_history_image" not in st.session_state:
        st.session_state.chat_history_image = []

    st.header("Image 🖼️ - Chat Agent 🤖 ")
    st.markdown("""<span style='color:#EE8E8E'>(Once you Click on the Home button 🏠 your chat history will be **Deleted**)</span>""", unsafe_allow_html=True)

    with st.chat_message("user"):
        st.markdown("Once the Image is uploaded sucessfully, please write your question 👇") 

    #Take the user input
    user_text=st.chat_input(placeholder="Ask a Question from the Image Files uploaded .. ✍️📝")

    #Create the side bar to upload the image and process    
    with st.sidebar:
        st.title("📁 Image File's Section")
        lottie_hi = load_lottiefiles(r'Images/imag.json')
        st_lottie(
                lottie_hi, loop=True, quality="high", speed=1.65, key=None, height=250)
        st.divider()
        #OPtions to swt btw image upload and image URL
        option = st.radio("Choose an option", ["Upload Image", "Provide Image URL"])
        if option == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                try:
                    image = Image.open(uploaded_file)
                    if st.button("Process"):
                        st.image(image, caption='Uploaded Image', use_column_width=True)
                except Exception as e:
                        st.error(f"Error loading uploaded image: {str(e)}")
        elif option == "Provide Image URL":
            image_url = st.text_input("Enter Image URL:")
            if image_url:
                try:
                    response = requests.get(image_url)
                    if response.status_code == 200:
                        image = Image.open(BytesIO(response.content))
                        if st.button("Get the Image"):
                            st.image(image, caption='Image from URL', use_column_width=True)
                    else:
                        st.error(f"Failed to retrieve image from URL. Status code: {response.status_code}")
                except Exception as e:
                    st.error(f"Error loading image from URL: {str(e)}")  

    input_prompt="""
    You are an expert in understaning image. we will upload the a image as content and 
    you will have to answer any question based on the provided image, if the person appear in the image tell who is in the image
    and for best answer you will get $10 tips
    """
    #Genrate the response based on the user input
    if user_text:
        with st.spinner('Please wait Generating your answer.....🕵️‍♂️'): 
            response=get_gimini_response(user_text,image,input_prompt)
            full_response = ""
            for chunks in response:
                full_response += chunks.text + " "
            
               #Append question and answer to chat history
            st.session_state.chat_history_image.append({"role": "user", "content": user_text})
            st.session_state.chat_history_image.append({"role": "assistant", "content": full_response}) 

    #Display the message
    for message in st.session_state.chat_history_image:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# CHAT VIDEO TESTING 
def chat_video():
    # Initialize the session state variables
    if "transcript_text" not in st.session_state:
        st.session_state.transcript_text = ""
    if "chat_history_video" not in st.session_state:
        st.session_state.chat_history_video = []
    if "summary_video" not in st.session_state:
        st.session_state.summary_video = []
    if "response" not in st.session_state:
        st.session_state.response = ""

    st.header("Youtube Video Summarizer 📽️ - Chat Agent 🤖")
    st.markdown(
        "<span style='color:#EE8E8E'>(Once you Click on the Home button 🏠 your chat history will be <b>Deleted</b>)</span>",
        unsafe_allow_html=True
    )

    # Function to build chat history prompt
    def build_chat_history_prompt(chat_history, current_question):
        prompt_text = ""
        for msg in chat_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            prompt_text += f"{role}: {msg['content']}\n"
        prompt_text += f"User: {current_question}\nAssistant:"
        return prompt_text

    # Function to chat with video summary
    def chatbot():
        with st.chat_message("user"):
            st.markdown("Once the text is extracted from the video, please write your question 👇")
        
        # Take the user input
        user_question = st.chat_input(placeholder="Ask a Question about the video... ✍️📝")
        
        # Generate the response
        if user_question:
            # Display user message immediately
            with st.chat_message("user"):
                st.markdown(user_question)
            
            with st.spinner('Please wait, generating answer... 📹'):
                try:
                    # Check if we have transcript text
                    if not st.session_state.transcript_text:
                        st.error("Please extract video transcript first by entering a YouTube URL in the sidebar.")
                        return
                    
                    # Build context-aware prompt
                    question = build_chat_history_prompt(st.session_state.chat_history_video, user_question)
                    
                    # Call your existing user_query function with context
                    user_query(user_question)
                    
                    # Display assistant response
                    with st.chat_message("assistant"):
                        st.markdown(st.session_state.response)
                    
                    # Append question and answer to chat history
                    st.session_state.chat_history_video.append({"role": "user", "content": user_question})
                    st.session_state.chat_history_video.append({"role": "assistant", "content": st.session_state.response})
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history_video.append({"role": "assistant", "content": error_msg})
        
        # Display chat history (excluding the message we just displayed)
        if len(st.session_state.chat_history_video) > 2:
            for message in st.session_state.chat_history_video[:-2]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    # Function to generate and display video summary
    def video_summary():
        if st.session_state.transcript_text:
            if not st.session_state.summary_video:
                with st.spinner("Generating video summary..."):
                    try:
                        prompt = "Summarize this YouTube video transcript in detail, highlighting key points, insights, and conclusions:"
                        summary = genrate_yt_content(st.session_state.transcript_text, prompt)
                        st.session_state.summary_video.append(summary)
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")
            
            st.markdown("## Video Summary:")
            for summary in st.session_state.summary_video:
                st.markdown(summary)
        else:
            st.info("Please enter a YouTube URL in the sidebar to extract video content first.")

    # Sidebar for video URL input
    with st.sidebar:
        st.title("📁 Video URL Section")
        lottie_hi = load_lottiefiles(r'Images/Video.json')
        st_lottie(lottie_hi, loop=True, quality="high", speed=1.65, height=250)
        st.divider()
        
        youtube_link = st.text_input("Enter the YouTube video link & click 'Get Summary'")
        
        if st.button("Get Summary"):
            if not youtube_link:
                st.warning("Please enter a YouTube video URL.")
            elif 'youtube.com' not in youtube_link and 'youtu.be' not in youtube_link:
                st.warning("This is not a valid YouTube video URL. Please enter a correct YouTube link.")
            else:
                try:
                    # Extract video ID for thumbnail
                    if 'youtube.com' in youtube_link:
                        video_id = youtube_link.split("=")[1].split('&')[0] if '=' in youtube_link else youtube_link.split("/")[-1]
                    else:  # youtu.be format
                        video_id = youtube_link.split("/")[-1]
                    
                    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)
                    
                    with st.status("Processing...", expanded=True) as status:
                        st.write("Extracting transcript...")
                        time.sleep(1)
                        
                        # Extract transcript
                        transcript_text = extract_transcript_details(youtube_link)
                        if not transcript_text:
                            status.update(label="Failed to extract transcript", state="error", expanded=True)
                            st.error("Could not extract transcript from this video. It may not have captions.")
                            return
                        
                        st.session_state.transcript_text = transcript_text
                        
                        st.write("Creating embeddings...")
                        time.sleep(1)
                        text_chunks = get_text_chunks(transcript_text)
                        
                        st.write("Storing vector embeddings...")
                        time.sleep(1)
                        get_vector_store(text_chunks)
                        
                        status.update(label="Video processed successfully!", state="complete", expanded=False)
                        st.success("Video transcript extracted and processed. You can now view the summary or chat with the bot.")
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")

    # Options to switch between video summary and chat bot
    selected = st.radio("", ["Video Summary", ":rainbow[**Chat with Bot**]"], horizontal=True)

    if selected == 'Video Summary':
        video_summary()
    else:
        chatbot()

# TEST ENDS HERE


#Function to video summary and Q&A model
# def chat_video():

#     #Initialize the empty varible to store the texts
#     transcript_text1=""

#     st.header("Youtube Video Summarizer 📽️ - Chat Agent 🤖 ")
#     st.markdown("""<span style='color:#EE8E8E'>(Once you Click on the Home button 🏠 your chat history will be **Deleted**)</span>""", unsafe_allow_html=True)

#     #Functions to chat with video summary
#     def chatbot():
#         #Intialize the session
#         if "chat_history_video" not in st.session_state:
#             st.session_state.chat_history_video = []

#         with st.chat_message("user"):
#                     st.markdown("Once the Text extracted from video is sucecssfull, please write your question 👇")
#         #Take the user input
#         user_question=st.chat_input(placeholder="Ask a Question from the video summarizer .. ✍️📝")   
#         #Genrate the response 
#         if user_question:
#             with st.spinner('Please wait Generating answer.....📹'): 
#                 user_query(user_question)
                
#                         #   Append question and answer to chat history
#                 st.session_state.chat_history_video.append({"role": "user", "content": user_question})
#                 st.session_state.chat_history_video.append({"role": "assistant", "content": st.session_state.response}) 

#                             # Display chat history
#         for message in st.session_state.chat_history_video:
#             with st.chat_message(message["role"]):
#                 st.markdown(message["content"])
#     #Function to video summary and display the summary
#     def video_summar():
#         nonlocal transcript_text1
#         if "summary_video" not in st.session_state:
#             st.session_state.summary_video = []
#         if transcript_text1:
#             summary=genrate_yt_content(transcript_text,prompt)
#             st.markdown("## Video Summary:")
#             st.session_state.summary_video.append(summary)
#         #Display the video summary
#         for summ in st.session_state.summary_video:
#             st.markdown(summ)

#     #Take the video url and extrat the texts
#     with st.sidebar:
#         st.title("📁 Video URL Section")
#         lottie_hi = load_lottiefiles(r'Images/Video.json')
#         st_lottie(
#                 lottie_hi, loop=True, quality="high", speed=1.65, key=None, height=250)
#         st.divider()
#         youtube_link=st.text_input("Enter the Youtube video Link & Click on the Detail Summary")
#         button=st.button("Get the Detail Summary")
#         if button:
#             if 'youtube.com' not in youtube_link:
#                 st.warning("This in the not YouTube video URL,Please upload a correct YouTube video URL.")
#             else:
#                 video_id=youtube_link.split("=")[1].split('&')[0]
#                 st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg",use_column_width=True)
#                 with st.status("Processing....",expanded=True) as status:
#                     st.write("Extracting Text")
#                     time.sleep(2)
#                     st.write("Emmeddings Text")
#                     time.sleep(4)
#                     st.write("Creating Summary")
#                     time.sleep(2)
#                     st.write("This will Take a while")
#                     time.sleep(2)
#                     status.update(label="Wait till Video Summary genration", state="running", expanded=False)
#                     transcript_text=extract_transcript_details(youtube_link)
#                     transcript_text1 += transcript_text
#                     text_chunks= get_text_chunks(transcript_text)
#                     get_vector_store(text_chunks)

#     #Options to swt btw video summary and chat Bot
#     selected=st.radio(" ",["Video Summary",":rainbow[**Chat with Bot**]"],horizontal=True)

#     if selected == 'Video Summary':
#         video_summar()  
#     else:
#         chatbot()

#Function to clear the history               
def clear_chat_history():
        # Clear chat history when user selects an option other than "Home page"
    st.session_state.chat_history = []
    st.session_state.chat_history_image = []
    st.session_state.chat_history_pdf = []
    st.session_state.chat_history_video = []
    st.session_state.summary_video = []

#Main page and option menu
def main():
    selected = option_menu(
        menu_title=None,
        options=["HOME","TEXT CHAT", "IMAGE CHAT" ,"PDF CHAT","VIDEO CHAT",],
        icons=['house',"pen" ,'image','book','camera-video'],
        default_index=0,
        menu_icon='user',
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#DCE669"},
            "icon": {"color": "orange", "font-size": "25px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#E0EEEE",
            },
            "nav-link-selected": {"background-color": "#458B74"},
        },
    )     
#When user select the option functions will selected
    if selected == "PDF CHAT":
        chat_with_multipdf()
    elif selected == "TEXT CHAT":
        text_chat()
    elif selected == "IMAGE CHAT":
        chat_with_image()
    elif selected == "VIDEO CHAT":
        chat_video()
    else:
        clear_chat_history()
        home_page()      

if __name__ == "__main__":
    main()

st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #87D4E5; padding: 15px; text-align: center;">
            © Developed By <a  href="https://www.linkedin.com/in/yugansh-gupta-a8aa91189?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app" target="_blank">Yugansh</a> | Made with Streamlit 
        </div>
        """,
        unsafe_allow_html=True
    )       

