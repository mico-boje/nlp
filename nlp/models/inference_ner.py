import os

from transformers import pipeline

from nlp.utils.utility import get_root_path

token_classifier = pipeline(
    "token-classification", 
    model=os.path.join(get_root_path(), 'data', 'models', 'xlm-roberta-base-ner'), 
    aggregation_strategy="simple"
)
print(token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn."))

da_text = """
De systemiske banker har fortsat en solid likviditets
situation. Alle de systemiske banker har en overle
velseshorisont med positiv overskudslikviditet på
mindst fem måneder i Nationalbankens følsomheds
analyse, hvor kundernes efterspørgsel efter likviditet
stiger, men bankerne ikke kan udstede ny nansie
ring.
"""
print(token_classifier(da_text))