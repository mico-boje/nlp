import os

from transformers import pipeline

from nlp.utils.utility import get_root_path

text = """
Europa-Kommissionen skal fremlægge et forslag til
revision a den nansielle lovgivning med okus på
rammerne for at føre makroprudentiel politik. Som
led heri skal Kommissionen fremlægge forslag, der
sikrer, at kapitalbufferne er tilstrækkeligt effektive og
dermed fungerer efter hensigten.
Kapitalbufferne, herunder den kontracykliske ka-
pitalbuffer, skal fungere som stødpude og afbøde
negative eekter ved en nansiel krise. De skal sikre,
at bankerne har tilstrækkeligt rum til at bære tab og
fortsat kan yde kredit til husholdninger og virksom-
heder i en krise.
På nuværende tidspunkt vil rigivelse a den kon-
tracykliske kapitalbuer i Danmark ikke medøre en
tilsvarende lettelse i kapitalkravene for danske banker,
da kapitalen anvendt til at opfylde den kontracykliske
kapitalbuer også anvendes til at opylde andre krav.1
De orventede positive eekter i orm a øget tabs- og
udlånskapacitet vil deror være kratigt reduceret.
Det er vigtigt at løse denne problematik og sikre de
makroprudentielle myndigheders handlerum.
"""
path = os.path.join(get_root_path(), 'data', 'models', 'google', 'mt5-small')
summarizer = pipeline("summarization", model=path, use_fast=False, max_length=100, min_length=30)
print(summarizer(text))
