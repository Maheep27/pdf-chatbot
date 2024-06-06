import os

import pandas as pd

import numpy as np

import pickle

import joblib

import re

import boto3

import string

import streamlit as st

import random

import tempfile

import logging

import torch

from transformers import AutoModelForSeq2SeqLM,AutoTokenizer, pipeline, BitsAndBytesConfig,T5Tokenizer, T5ForConditionalGeneration
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import streamlit.components.v1 as components  # Import Streamlit

from streamlit_chat import message

from sentence_transformers import SentenceTransformer, util

from datetime import datetime

from langchain import HuggingFaceHub

from langchain.prompts import PromptTemplate

from langchain.chains import LLMChain

from langchain.llms import HuggingFacePipeline

from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


from random import randrange

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer

from .config import config
from PIL import Image

from sklearn.metrics.pairwise import cosine_similarity



import datetime

now = datetime.date.today()

from datetime import datetime

 

 

#to find path of current working directory   

absolute_path = os.getcwd()

full_path = os.path.join(absolute_path, config['logging_foler'])

 

 

# Check whether the specified path exists or not

isExist = os.path.exists(full_path)

if not isExist:

   # Create a new directory because it does not exist

   os.makedirs(full_path)

 

file_name = str(full_path)+"qna_" + str(now) + ".log"

 

class ExcludeLogLevelsFilter(logging.Filter):

    def __init__(self, exclude_levels):

        self.exclude_levels = set(exclude_levels)


    def filter(self, record):

        return record.levelname not in self.exclude_levels

 

    

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)  # Set the desired logging level, e.g., INFO, DEBUG, WARNING, ERROR, etc.

 

log_filename = 'custom_filename.log'  # Specify your custom filename here

# Create a file handler with the custom filename

file_handler = logging.FileHandler(file_name)

 

# Create a JSON formatter to format log records as JSON

json_formatter = logging.Formatter('{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}')

 

# Set the JSON formatter for the file handler

file_handler.setFormatter(json_formatter)

# Add the file handler to the logger

logger.addHandler(file_handler)

# Exclude log levels DEBUG, WARNING, and ERROR from being saved to the file

exclude_levels = ['DEBUG', 'WARNING', 'ERROR']

file_handler.addFilter(ExcludeLogLevelsFilter(exclude_levels))

 

 

 

class Qna:


    def __init__(self): 

        """

        load all the static files from s3

        """


        self.flan_chain=self.load_generative()
        self.embedding = HuggingFaceEmbeddings(model_name=config['embedding_model'])
        self.username=config['name']


 

    # @st.cache_resource

    def load_generative(self):

        model=config['LLM_model']

        device=config['LLM_gpu']

        # quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

 

        FT_MODEL = AutoModelForSeq2SeqLM.from_pretrained(model,device_map='auto')

        FT_MODEL_TOKENIZER = AutoTokenizer.from_pretrained(model)

 

        llm_pipeline =  pipeline(

                            task=config['LLM_task'],

                            model=FT_MODEL,

                            temperature=config['LLM_temp'],

                            max_length= config['LLM_max_length'],

                            min_length=config['LLM_min_length'],

                            no_repeat_ngram_size= config['LLM_no_repeat_ngram'],

                            tokenizer=FT_MODEL_TOKENIZER)

 

        llm=HuggingFacePipeline(pipeline=llm_pipeline)

        prompt = PromptTemplate(template="Generate answers in various steps as detailed as possible of the given question deliminted by triple backticks based on context given in angle bracket <{context}>   ```{question}```", input_variables=["context","question"])

 

        chain = LLMChain(llm=llm, prompt=prompt)

        return chain

    # Function to save uploaded file
    def save_uploaded_file(self,uploaded_file, save_directory):
        file_path = os.path.join(save_directory, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    # Function to create Faiss index from PDF
    def create_faiss_index(self,file_path):
        global faiss_index
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=config['chunk_size'], chunk_overlap=config['chunk_overlap'], separator=" ")
        pages = text_splitter.split_documents(pages)

        faiss_index = FAISS.from_documents(pages, self.embedding)
        faiss_index.save_local(self.username)

    # Function to retrieve context from Faiss index
    def get_context(self,question):
        global faiss_index
        docs = faiss_index.similarity_search(question, k=config['number_of_documents'])
        return docs[0].page_content + docs[1].page_content  # Combine top 2 similar documents for context

    def chat_generative(self,question,model):

        """
        Answer a question based on the most similar context from the dataframe texts
        """

        

        flan_chain = self.flan_chain


        context=self.get_context(question)

 
        examples = [

            {

                "context": f"""{context}""",
                "question": f"""{question}?"""

            }

 

        ]


        if model=="Flan-T5-xxl":

            predictions = flan_chain.apply(examples)
            logger.info("context searched on vector db for flan-T5 model: "+str(predictions))

        else:

            return "Please use flan-t5-model"
            logger.info("context searched on vector db for llama-2 model: "+str(predictions))


        return predictions[0]['text']

 

 

    # def chat_(self,index=None, question=None):

    #     return "Test Result"

 

    def generate_ans(self,user_input,model):

        print('generating answer...')

        result1 = self.chat_generative(question = user_input,model=model)

        logger.info("Answer generated by generative LLM algorithm : "+str(result1))

        # return st.session_state.bot.append(result1)

        return result1


 

    def main(self,query,model):

        try:

            logger.info("Question asked by client : "+str(query))

            logger.info("model choosen by User : "+str(model))

           

            answer = self.generate_ans(query,model=model)

            logger.info("Answer generated by bot : "+str(answer))

            return answer


        except Exception as e:  

            logger.error("Error inside main function of process file : "+str(e))

            return None