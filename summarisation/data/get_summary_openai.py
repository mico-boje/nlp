import os
import openai


from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain


LLM = OpenAI()

PROMPT_TEMPLATE = """Write a concise summary in circa 200 words of the following:

{text}

CONCISE SUMMARY IN DANISH:"""
PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["text"])

def get_summary(pages):
    try: 
        docs = [Document(page_content=t) for t in pages]
        chain = load_summarize_chain(LLM, chain_type="map_reduce", return_intermediate_steps=False, map_prompt=PROMPT, combine_prompt=PROMPT)
        result = chain({"input_documents": docs}, return_only_outputs=True)
        return result["output_text"]
    except openai.error.InvalidRequestError:
        chunked_pages = []
        summarized_pages = []
        for e, _ in enumerate(pages):
            if e > len(pages) - 1:
                break
            chunked_pages.append(pages[e:e+4])
            e+=4
        for page in chunked_pages:
            docs = [Document(page_content=page)]
            chain = load_summarize_chain(LLM, chain_type="map_reduce", return_intermediate_steps=False, map_prompt=PROMPT, combine_prompt=PROMPT)
            result = chain({"input_documents": docs}, return_only_outputs=True)
            summarized_pages.append(result["output_text"])
        get_summary(summarized_pages)
            
