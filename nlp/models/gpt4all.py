import re

from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class GPT4AllInference:
    def __init__(self, model_path, max_tokens=512,**kwargs):
        self._callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.llm = GPT4All(
            model=model_path, 
            callback_manager=self._callback_manager, 
            verbose=False,
            n_ctx=max_tokens,
            **kwargs)
        
    def __call__(self, prompt: PromptTemplate, **kwargs):
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        tags = llm_chain.run(**kwargs)
        print(tags)
        return self._extract_tags(tags)
    
    def _extract_tags(self, text):
        # Find Answer: and only take the text after that
        tags_str = re.search(r'Answer:\s*(.*)', text).group(1)
        # Remove the <result> and </result> tags
        tags_str = tags_str.replace('<result>', '').replace('</result>', '').replace('.', '')
        # Remove the brackets from the tags string
        tags_str = tags_str.replace('[', '').replace(']', '')
        # Remove Tags = from the tags string
        tags_str = tags_str.replace('Tags = ', '')
        # Split the tags string into individual tags
        tags_list = tags_str.split(', ')
        # Strip any whitespace from the individual tags
        tags_list = [tag.strip() for tag in tags_list]
        return tags_list
        