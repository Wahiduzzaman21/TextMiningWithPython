import nltk
import spacy
# Tokenize with NLTK
sentence = "It's too cold to go outside, we'd be better watering our neighbour's plant tomorrow"
print(nltk.word_tokenize(sentence, 'english', False))
# Tokenize with Spacy
nlp_en = spacy.load("en_core_web_sm")
doc = nlp_en(sentence)
print([x.text for x in doc])

# POS tagging with NLTK (Using Penn Treebank Tagset)
tokens = nltk.word_tokenize(sentence, 'english', False)
print(nltk.pos_tag(tokens))
# POS tagging with Spacy (Universal Dependency Tagset)
print([(x.text, x.pos_)for x in doc])

