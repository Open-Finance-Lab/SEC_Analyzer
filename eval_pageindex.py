import json
import asyncio
import openai
import pageindex.utils as utils
import os, requests
from pageindex import PageIndexClient

import pprint


# Get your PageIndex API key from https://dash.pageindex.ai/api-keys
PAGEINDEX_API_KEY = os.environ.get('PAGEINDEX_API_KEY')
pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')



async def call_llm(prompt, model="gpt-5", temperature=1):
    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

# You can also use our GitHub repo to generate PageIndex tree
# https://github.com/VectifyAI/PageIndex

'''
def get_pdfs():
    doc_id = pi_client.submit_document(pdf_path)["doc_id"]
    print('Document Submitted:', doc_id)
    
    return doc_id
'''

def get_tree(doc_id):
    if pi_client.is_retrieval_ready(doc_id):
        tree = pi_client.get_tree(doc_id, node_summary=True)['result']
    
    return tree

async def retrieving(tree, question):

    tree_without_text = utils.remove_fields(tree.copy(), fields=['text'])

    search_prompt = f"""
    You are given a question and a tree structure of a document.
    Each node contains a node id, node title, and a corresponding summary.
    Your task is to find all nodes that are likely to contain the answer to the question.

    Question: {question}

    Document tree structure:
    {json.dumps(tree_without_text, indent=2)}

    Please reply in the following JSON format:
    {{
        "thinking": "<Your thinking process on which nodes are relevant to the question>",
        "node_list": ["node_id_1", "node_id_2", ..., "node_id_n"]
    }}
    Directly return the final JSON structure. Do not output anything else.
    """

    tree_search_result = await call_llm(search_prompt)
    
    node_map = utils.create_node_mapping(tree)
    tree_search_result_json = json.loads(tree_search_result)

    print('Reasoning Process:')
    utils.print_wrapped(tree_search_result_json['thinking'])

    print('\nRetrieved Nodes:')
    for node_id in tree_search_result_json["node_list"]:
        node = node_map[node_id]
        print(f"Node ID: {node['node_id']}\t Page: {node['page_index']}\t Title: {node['title']}")
    
    return tree_search_result, node_map, tree_without_text
        
async def get_answers(tree_search_result, node_map, question):      
    node_list = json.loads(tree_search_result)["node_list"]
    relevant_content = "\n\n".join(node_map[node_id]["text"] for node_id in node_list)
    answer_prompt = f"""
    Answer the question based on the context:

    Question: {question}
    Context: {relevant_content}

    Provide a clear, concise answer based only on the context provided.
    """

    answer = await call_llm(answer_prompt)

    return answer

def main():
    question = "What is the quantity of restructuring costs directly outlined in Pepsico's income statements for FY2022? If restructuring costs are not explicitly outlined then state 0."
    
    #"pi-cmga9fom101eo09r3pm9uhsty"
    doc_id = "pi-cmgbg6hav00fe0aqsraygupct"
    tree = get_tree(doc_id) 


    
    tree_search_result, node_map, tree_without_text = asyncio.run(retrieving(tree, question))
        
    answer = asyncio.run(get_answers(tree_search_result, node_map, question))
    print('\nFinal Answer:')
    utils.print_wrapped(answer)
    
    
if __name__ == "__main__":
    main()
'''   
##############################################################################
# DEBUGGING NODE MAP
##############################################################################  
    node_map_debug_path = "tree_search_result_debug.json"
    with open(node_map_debug_path, "w", encoding="utf-8") as f:
        json.dump(tree_search_result, f, indent=2, ensure_ascii=False)
    print(f"[DEBUG] Node map saved to {node_map_debug_path}")
''' 