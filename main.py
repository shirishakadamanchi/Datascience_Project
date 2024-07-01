import os
import urllib.request
import torch
import transformers
import pandas as pd
import pdfkit
from bs4 import BeautifulSoup
from torch import cuda, bfloat16
from langdetect import detect_langs
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain.text_splitter import RecursiveCharacterTextSplitter
import boto3
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
# from beautifulsoup4 import bs4
import streamlit as st


from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

model_name = os.environ["model_id"]
hf_auth = os.environ["hf_auth"]

@st.cache_resource
class ModelLoader:
    """
    Model loading class for both model_id and model tokenizer
    """
    def __init__(self, model_id, hf_auth):
        self.model_id = model_id
        self.hf_auth = hf_auth
        self.device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
        self.model = self.initialize_model()
        self.tokenizer = self.load_tokenizer()

    def initialize_model(self):
        # Set quantization configuration
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype='bfloat16'
        )

        # Initialize and load the model
        model_config = AutoConfig.from_pretrained(
            self.model_id,
            use_auth_token=self.hf_auth,
            cache_dir='./model/'
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            cache_dir='./model/',
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
            use_auth_token=self.hf_auth  
        )

        # Enable evaluation mode
        model.eval()

        return model
    

    def load_tokenizer(self):
        # Initialize and load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            use_auth_token=self.hf_auth
        )
        return tokenizer
# loading the model
model_var = ModelLoader(model_id=model_name, hf_auth=hf_auth)
tokenizer = model_var.tokenizer
model = model_var.model
device = model_var.device


# creating chunks
stop_list = ['\nHuman:', '\n```\n']
stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:

        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

# language function
def detect_language_with_langdetect(line):
    try:
        langs = detect_langs(line)
        for item in langs:
            # The first one returned is usually the one that has the highest probability
            return item.lang, item.prob
    except: return "err", 0.0

# function to translate code
def translate_text(text, source_language_code, target_language_code):
    # Create a client
    translate = boto3.client(service_name='translate', region_name='ap-south-1', aws_access_key_id="keyid", aws_secret_access_key="accesskey")

    # Translate text
    result = translate.translate_text(Text=text,
                                      SourceLanguageCode=source_language_code,
                                      TargetLanguageCode=target_language_code)

    # Return the translated text
    return result.get('TranslatedText')

# convert DF to html
def dataframe_to_html(df):
    return df.to_html(escape=False)

def dataframe_to_html_with_newlines(df):
    df_html = df.to_html(escape=False)
    df_html = df_html.replace("\\n", "<br>")
    df_html = df_html.replace("\\n", "<br>")
    return df_html

def remove_image_tags(html_content):
    # Parse the HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all image tags
    for img_tag in soup.find_all('img'):
        # Remove each image tag
        img_tag.decompose()

    # Return the modified HTML
    return str(soup)

# Function to display DataFrame with HTML content
def show_dataframe_with_html(df):
    for _, row in df.iterrows():
        # Assuming 'Name' is plain text and 'Profile' contains HTML
        st.write(f"Data_Points: {row['Data_Points']}")
        st.markdown(row["Value"], unsafe_allow_html=True)


stopping_criteria = StoppingCriteriaList([StopOnTokens()])

generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model rambles during chat
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # max number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)
llm = HuggingFacePipeline(pipeline=generate_text)


## stream lit application
st.title("T.H.O.R")
st.sidebar.title("Store URLs")

urls = []
for i in range(3):
    if i == 0:
        url = st.sidebar.text_input(f"Steam")
    elif i == 1:
        url = st.sidebar.text_input(f"Xbox")
    elif i == 2:
        url = st.sidebar.text_input(f"Playstation")
    if url != '':
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")


main_placeholder = st.empty()

if process_url_clicked:
    # load data
    url_str = urls[0].split('/')[-2]
    pdfkit.from_url(urls[0], f'./pdfs/{url_str}.pdf') # saved the url to PDF
    main_placeholder.text("Url Snapshot complete...✅✅✅")

    # Conditioning File
    loader = PyPDFLoader(f'./pdfs/{url_str}.pdf')

    main_placeholder.text("Detecting Language...✅✅✅")
    # Detecting language
    lang = detect_language_with_langdetect(loader.load()[0].page_content)[0]

    #Load the document by calling loader.load()
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_splits = text_splitter.split_documents(documents)

    ## Creating a Vectorstore
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cuda"}
    main_placeholder.text("Model Initiation...✅✅✅")
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    # storing embeddings in the vector store
    vectorstore = FAISS.from_documents(all_splits, embeddings)

    main_placeholder.text("Creating Knowledge Base...✅✅✅")
    # Creating Chain
    chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)

    # main_placeholder.text("Data Loading...✅✅✅")

    prompts_list_en = ["Give me Title, Developer, Publisher, Release Date, Genre, Theme, User Rating, Age Rating, Franchise and Game Modes of the game in structured format",
                       "Get the Minimum system requirements",
                       "Get the Recommended system requirements",
                   "Does the game have Steam Trading Cards, give me simple yes or no",
                   "Does the game have Steam Cloud, give me simple yes or no",
                   "Did the game win any awards, give me simple yes or no"
                   ]
    main_placeholder.text("Querying Url Snapshot...✅✅✅")
    if lang != "en":
        prompts_list = [translate_text(prompt,"en",lang) for prompt in prompts_list_en]
    else:
        prompts_list = prompts_list_en

    PromptAnswers = []
    for q in prompts_list:
        result = chain({"question": q, "chat_history":[]})
        PromptAnswers.append(result['answer'])
    main_placeholder.text("Completed Knowledge Base Generation...✅✅✅")

    ans_lis = PromptAnswers[0].split("\n")
    ans_lis = ans_lis[1:-1]
    ans_lis = [ele for ele in ans_lis if ele]
    ans_dict = {item.split(':')[0].strip(): item.split(':')[1].strip() for item in ans_lis}
    main_placeholder.text("Generating output...✅✅✅")

    if len(PromptAnswers[2].split(",")) > 1:
        Trading_val =  PromptAnswers[3].split(",")[0]
    else:
        Trading_val = PromptAnswers[3]

    if len(PromptAnswers[4].split(",")) > 1:
        Steam_Cloud_val =  PromptAnswers[4].split(",")[0]
    else:
        Steam_Cloud_val = PromptAnswers[4]    

    ans_dict.update({"Minimum System Requirements":PromptAnswers[1]})
    ans_dict.update({"Recommended System Requirements":PromptAnswers[2]})
    ans_dict.update({"Steam Trading Cards":Trading_val})
    ans_dict.update({"Supports Steam Cloud":Steam_Cloud_val})
    ans_dict.update({"Awards":PromptAnswers[5]})

    opener = urllib.request.FancyURLopener({})
    url = urls[0]
    f = opener.open(url)
    content = f.read()

    soup = BeautifulSoup(content, 'html.parser') 
    mydivs = soup.find_all("div", {"class": "game_area_description", "id" : "game_area_description"})

    content_0 = str(mydivs[0])

    # Remove image tags
    modified_html = remove_image_tags(content_0)

    ssoup = BeautifulSoup(modified_html)
    prettyHTML = ssoup.prettify()

    ans_dict.update(({'Description':prettyHTML}))

    result_df = pd.DataFrame(ans_dict.items(), columns=["Data_Points","Value"])
    html = dataframe_to_html_with_newlines(result_df)
    
    try:
        st.markdown(html, unsafe_allow_html=True)
        # show_dataframe_with_html(result_df)
    except:
        st.write(result_df)

# query = main_placeholder.text_input("Question: ")
# if query:
#     if os.path.exists(file_path):
#         with open(file_path, "rb") as f:
#             vectorstore = pickle.load(f)
#             chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
#             result = chain({"question": query}, return_only_outputs=True)
#             # result will be a dictionary of this format --> {"answer": "", "sources": [] }
#             st.header("Answer")
#             st.write(result["answer"])
            
#             df = pd.DataFrame(np.random.randn(50, 2), columns=("datapoints", "Information"))

#             st.write(df)
#             # # Display sources, if available
#             # sources = result.get("sources", "")
#             # if sources:
#             #     st.subheader("Sources:")
#             #     sources_list = sources.split("\n")  # Split the sources by newline
#             #     for source in sources_list:
#             #         st.write(source)

