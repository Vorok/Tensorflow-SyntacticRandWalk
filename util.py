import spacy
nlp = spacy.load('en')
def clean_up(text):  # clean up your text and generate list of words for each document. 
	removal=['ADV','PRON','CCONJ','PUNCT','PART','DET','ADP','SPACE']
	text_out = []
	doc= nlp(text)
	for token in doc:
		if token.is_stop == False and token.is_alpha and len(token)>2 and token.pos_ not in removal:
			lemma = token.lemma_
			text_out.append(lemma)
	return text_out