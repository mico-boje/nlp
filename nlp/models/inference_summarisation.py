import os

from transformers import pipeline

from nlp.utils.utility import get_root_path

text = """
WASHINGTON — Problems with the T-7A Red Hawk, including a potentially dangerous escape system and ejection seat, 
have caused the U.S. Air Force to push back a production decision and deliveries of the service's next jet trainer aircraft.
"""
path = os.path.join(get_root_path(), 'data', 'models', 't5-small')
summarizer = pipeline("summarization", model=path, use_fast=False, max_length=20, min_length=1)
print(summarizer(text))
