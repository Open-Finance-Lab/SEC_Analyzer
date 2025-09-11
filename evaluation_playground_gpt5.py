import os
import sys
import json
import pickle
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# LangChain Stuff -  langchain_community
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

# LangChain Model Wrappers -  langchain_community
from langchain_community.chat_models import ChatOpenAI

# Model Providers
import openai
import tiktoken

openai.api_key = os.environ.get('OPENAI_API_KEY', 'dummy-key')

##############################################################################
# MODEL CONFIGS
##############################################################################
configs = [
            {"provider": "openai",     "model_name":"gpt-5",   "eval_mode":"singleStore",        "temp":None,   "max_tokens":2048},
            {"provider": "openai",     "model_name":"gpt-5",   "eval_mode":"sharedStore",        "temp":None,   "max_tokens":2048},
            {"provider": "openai",     "model_name":"gpt-5-mini",   "eval_mode":"inContext",          "temp":None,   "max_tokens":2048},
            {"provider": "openai",     "model_name":"gpt-4o",   "eval_mode":"inContext_reverse",  "temp":None,   "max_tokens":2048},
            {"provider": "openai",     "model_name":"gpt-5",   "eval_mode":"oracle",             "temp":None,   "max_tokens":2048},
            {"provider": "openai",     "model_name":"gpt-5",   "eval_mode":"oracle_reverse",     "temp":None,   "max_tokens":2048},
            {"provider": "openai",     "model_name":"gpt-5",   "eval_mode":"closedBook",         "temp":None,   "max_tokens":2048},
]


##############################################################################
# DATASET CONFIG
##############################################################################
PATH_CURRENT = os.path.abspath(os.getcwd())
PATH_DATASET_JSONL = PATH_CURRENT + "/data/financebench_open_source.jsonl"
PATH_DOCUMENT_INFO_JSONL = PATH_CURRENT + "/data/financebench_document_information.jsonl"
PATH_RESULTS = PATH_CURRENT + "/results/"
PATH_PDFS = PATH_CURRENT + "/pdfs/"

# Choose DATASET PORTION:
# - ALL: Full Dataset
# - OPEN_SOURCE: Open Source Part (n=150)
# - CLOSED_SOURCE: Closed Source Part --> Request access at contact@patronus.ai
DATASET_PORTION = "OPEN_SOURCE"   

##############################################################################
# VECTOR STORE SETUP
##############################################################################
VS_CHUNK_SIZE = 1024
VS_CHUNK_OVERLAP = 30
VS_DIR_VS = PATH_CURRENT + "/vectorstores"

##############################################################################
# LOAD DATASET
##############################################################################

# Load Full Dataset 
df_questions = pd.read_json(PATH_DATASET_JSONL, lines=True)
df_meta = pd.read_json(PATH_DOCUMENT_INFO_JSONL, lines=True)
df_full = pd.merge(df_questions, df_meta, on="doc_name")


df_questions = df_questions.iloc[100:150]


# Get all docs
df_questions = df_questions.sort_values('doc_name')
ALL_DOCS = df_questions['doc_name'].unique().tolist()
print(f"Total number of distinct PDF: {len(ALL_DOCS)}")

# Select relevant dataset portion
if DATASET_PORTION != "ALL":
    df_questions = df_questions.loc[df_questions["dataset_subset_label"]==DATASET_PORTION]
print(f"Number of questions: {len(df_questions)}")

# Check relevant documents
df_questions = df_questions.sort_values('doc_name')
docs = df_questions['doc_name'].unique().tolist()
print(f"Number of distinct PDF: {len(docs)}")

##############################################################################
# HELPER FUNCTIONS (PDF-PARSING + VECTOR-STORE SETUPS)
##############################################################################
def get_pdf_text(doc):
    
    path_doc = f"{PATH_PDFS}/{doc}.pdf"
    pdf_reader = PyMuPDFLoader(path_doc)
    pdf_text = pdf_reader.load()

    return pdf_text

def build_vectorstore_retriever(docs, embeddings = None):
    if embeddings is None:
        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get('OPENAI_API_KEY', 'dummy-key'))

    if docs == "all":
        docs = ALL_DOCS
        db_path = VS_DIR_VS + "/shared"
    else:
        docs = [docs]
        db_path = VS_DIR_VS + "/" + docs[0]
    
    # Create Vector Store if not already existing
    if not os.path.exists(db_path):
        
        # Create folder for vector store
        os.mkdir(db_path) 

        # Create vector store itself --> chrom.sqlite3 database
        if not os.path.exists(f"{db_path}/chroma.sqlite3"):
            vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)
            vectordb.persist()
    
            # Add Documents to Vector store    
            for doc in docs:
                pdf_text = get_pdf_text(doc)
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size = VS_CHUNK_SIZE,
                    chunk_overlap = VS_CHUNK_OVERLAP,
                )
                splitted_texts = text_splitter.split_documents(pdf_text)
        
                # Add to vector store
                vectordb.add_documents(documents=splitted_texts)
                vectordb.persist()

    else:
        vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)

    return vectordb.as_retriever(), vectordb

##############################################################################
# MODEL + CALL HANDLERS
##############################################################################

def get_max_context_length(prompt, openai_cutoff=105000):

    # (1) Check OpenAI Tokenizer
    tokenizer_openai = tiktoken.encoding_for_model("gpt-4-1106-preview")
    tokens_openai = tokenizer_openai.encode(prompt)
    nb_tokens_openai = len(tokens_openai)
    number_of_chars_openai = len(prompt)

    if nb_tokens_openai > openai_cutoff:
        tokens_openai_tokens = [tokenizer_openai.decode_single_token_bytes(token) for token in tokens_openai]
        token_lengths_openai = [len(token) for token in tokens_openai_tokens]
        number_of_chars_openai = sum(token_lengths_openai[:openai_cutoff])

    return number_of_chars_openai

def get_model(provider="openai", model_name="gpt-5-mini", temp=None, max_tokens=2048):

    if provider == "openai":
        if model_name == "gpt-5-mini":
            return ChatOpenAI(
                model_name=model_name, 
                temperature=1,
                model_kwargs={"max_completion_tokens": max_tokens}
                )
        else:
            if temp is None:
                return ChatOpenAI(
                    model_name=model_name, 
                    max_tokens=max_tokens
                    )
            else:
                return ChatOpenAI(
                    model_name=model_name, 
                    temperature=temp, 
                    max_tokens=max_tokens
                    )
        
    else:
        return None


def get_answer(model, eval_mode, question, context, retriever, retriever_only=False):

    retrieved_documents = []

    if eval_mode == "closedBook":
        prompt = f"Answer the question while showing your reasoning step by step before giving the final answer. Format your response with two fields: reasoning and final_answer. Answer this question: {question}"
        answer = model.predict(prompt)
        
    elif eval_mode == "oracle":
        prompt = f"Answer this question: {question} \nHere is the relevant evidence that you need to answer the question:\n[START OF FILING] {context} [END OF FILING]"
        answer = model.predict(prompt)

    elif eval_mode == "oracle_reverse":
        
        prompt = f"Context:\n[START OF FILING] {context} [END OF FILING\n\n Answer this question: {question} \n"
        answer = model.predict(prompt)

    elif eval_mode in ["inContext",  "inContext_reverse"]:
        
        '''
        # Context Cutoff to satisfy max tokens
        context = context[:100000]
        '''
         # Context Cutoff to satisfy max tokens
        max_number_of_chars = get_max_context_length(context)
        context = context[:max_number_of_chars]
        
        if eval_mode == "inContext":
            prompt = f"Answer this question: {question} \nHere is the relevant filing that you need to answer the question:\n[START OF FILING] {context} [END OF FILING]"
        else:
            prompt = f"Context:\n[START OF FILING] {context} [END OF FILING]\n\n Answer this question: {question}\n"

        answer = model.predict(prompt)

    elif eval_mode == "singleStore" or eval_mode == "sharedStore":
        
        # Retrieval-only mode if model=None (No LLM calls, only queries in VectorDB)
        if not model:           
            prompt = f"{question}"
            s = retriever.invoke(prompt)
            return ("", s)

        else:

            # Don't add a question prefix as RetrievalQA will do some automatic prompt wrapping
            # --> This can replace by more advanced Retrieval Strategies
            prompt = f"{question}"
            qa = RetrievalQA.from_chain_type(
                llm=model,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
            )
            s = qa(prompt)
            
            answer = s["result"]
            retrieved_documents = s["source_documents"]


    
    return (answer, retrieved_documents)

def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='FinanceBench GPT-5 Evaluation')
    parser.add_argument('--eval-mode', choices=['singleStore', 'sharedStore', 'inContext', 'inContext_reverse', 'oracle', 'oracle_reverse', 'closedBook'],
                       default='oracle', help='Evaluation mode')
    
    args = parser.parse_args()
    
    # Find the config for the specified eval_mode
    model_config = None
    for config in configs:
        if config["eval_mode"] == args.eval_mode:
            model_config = config
            break
    
    if model_config is None:
        print(f"Error: No config found for eval_mode: {args.eval_mode}")
        return
    
    # Set evaluation questions
    df_eval = df_questions

    # Get the model
    model = get_model(provider=model_config["provider"],
                      model_name=model_config["model_name"],
                      temp=model_config["temp"],
                      max_tokens=model_config["max_tokens"])

    print(f"--> Evaluating: {model_config['model_name']} / {model_config['eval_mode']}")

    last_docs = None
    results = []

    # Run evaluation on the model  --> Sort along doc_name to reuse retriever configs in memory
    for k, (idx, row) in tqdm(enumerate(df_eval.sort_values("doc_name").iterrows()), total=len(df_eval)):
            
        
        # (A) Setup Context or Retriever
        if model_config["eval_mode"] == "closedBook":
            retriever = None
            context = ""
        
        elif model_config["eval_mode"] in ["inContext", "inContext_reverse"]:
            retriever = None
            docs = row["doc_name"]
            if not (last_docs == docs):
                pages = get_pdf_text(row["doc_name"])
                context = "\n\n".join([page.page_content for page in pages])
                
        
        elif model_config["eval_mode"] in ["oracle", "oracle_reverse"]:
            context = "\n\n".join([evidence["evidence_text_full_page"] for evidence in row["evidence"]])
            retriever = None

        elif model_config["eval_mode"] in ["singleStore", "sharedStore"]:
            context = ""
            docs = "all"

            if model_config["eval_mode"] == "singleStore":
                docs = row["doc_name"]
            
            if not (last_docs == docs):
                retriever, _ = build_vectorstore_retriever(docs=docs)
                last_docs = docs


        else:
            raise ValueError("Unknown 'eval_mode'!")


        # (B) Model Call
        (answer, retrieved_documents) = get_answer(
                                            model=model, 
                                            eval_mode=model_config["eval_mode"], 
                                            question=row["question"], 
                                            context=context, 
                                            retriever=retriever
                                            )

        

        # (C) Bookkeeping
        results.append({
                            **model_config, 
                            "financebench_id" : row["financebench_id"],
                            "question" : row["question"],
                            "gold_answer": row["answer"],
                            "model_answer": answer,
                            "retrieved_documents" : retrieved_documents,
                        })
    
    os.makedirs(PATH_RESULTS, exist_ok=True)
    output_file = f"{PATH_RESULTS}/{model_config['model_name']}_{model_config['eval_mode']}_t4.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
        # 构建符合要求的JSON格式
            json_record = {
                "financebench_id": result["financebench_id"],
                "model_name": result["model_name"],
                "eval_mode": result["eval_mode"],
                "temp": result["temp"],
                "question": result["question"],
                "gold_answer": result["gold_answer"],
                "model_answer": result["model_answer"],
                }   
            f.write(json.dumps(json_record, ensure_ascii=False) + '\n')

    """
    os.makedirs(PATH_RESULTS, exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results.to_csv(PATH_RESULTS + "/" + model_config["model_name"] + "_" + model_config["eval_mode"] + ".csv")
    """
    print(f"Results saved to: {PATH_RESULTS}/{model_config['model_name']}_{model_config['eval_mode']}.jsonl")

if __name__ == "__main__":
    main()
