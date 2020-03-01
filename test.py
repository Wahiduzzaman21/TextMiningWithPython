import string
import regex

# Remove Punctuation

sentence = "Feeling loved, even when I'm sick🍫☕💓#likeforlike #chocolate #bf #iloveyou #aftereight #couplegoals"
translator = str.maketrans('', '', string.punctuation)
finalResult = sentence.translate(translator)
print(finalResult)

# Remove emoji using regex
emoPattern = regex.compile("""\p{So}\p{Sk}*""")
outputSentence1 = emoPattern.sub(' ',finalResult)
print(outputSentence1)

# Remove emoji using encode and decode ascii by ignoring
outputSentence2 = finalResult.encode('ascii', 'ignore').decode("utf-8")
print(outputSentence2)

# Remove URL from sentence using regex

subject = "Omg, check out these fabulous shoes https://thiswebsitedoesntexistsodontbother.com/omgshoesss yes they can be yours"
result = regex.sub(r"http\S+", "", subject)
print(result)

from nltk.corpus import stopwords
stop_en = stopwords.words("English")
input = ["i","have","a","cat","named","mr","whiskers","he","is","a","very","hungry","cat"]
input = [x for x in input if not x in stop_en]
print(input)

from bnltk.tokenize import Tokenizers
t = Tokenizers()

# Toeknize bangla word
sentenceBangla = "প্রধানমন্ত্রী শেখ হাসিনা বলেছেন, অর্থনৈতিকভাবে বাংলাদেশ এখন সিঙ্গাপুরের চেয়েও শক্তিশালী।"
print(t.bn_word_tokenizer(sentenceBangla))
inputBangla = t.bn_word_tokenizer(sentenceBangla)

# Stemming bangla word
from bnltk.stemmer import BanglaStemmer
bn_stemmer = BanglaStemmer()
outputStem = [x for x in inputBangla ]
for x in outputStem:
    print(bn_stemmer.stem(x))


import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

# Pos tagging bangla word
from bnlp.bengali_pos import BN_CRF_POS
bn_pos = BN_CRF_POS()
model_path = "F:/MSCS/Personal/Python/TextMiningWithPython/model/bn_pos_model.pkl"
res = bn_pos.pos_tag(model_path, "প্রধানমন্ত্রী শেখ হাসিনা বলেছেন অর্থনৈতিকভাবে বাংলাদেশ এখন সিঙ্গাপুরের চেয়েও শক্তিশালী")
print(res)

#ngrams
from nltk import ngrams
text = "প্রধানমন্ত্রী শেখ হাসিনা বলেছেন অর্থনৈতিকভাবে বাংলাদেশ এখন সিঙ্গাপুরের চেয়েও শক্তিশালী"
n = 2
bigrams = ngrams(text.split(),n)
for grams in bigrams:
    print(grams)

#ngrams another approach by chunk and rule

from nltk import RegexpParser
chunktriGram = r"""NC: {<NC><NC><NC>}"""
chunkparsertrigram = RegexpParser(chunktriGram)

chunked = chunkparsertrigram.parse(res)

for subtree in chunked.subtrees():
    if subtree.label() == 'NC':
        leaves = subtree.leaves()
        wholetext=""
        for leaftext in leaves:
            wholetext = wholetext+leaftext[0]+" "
        print(wholetext)

from nltk.corpus import inaugural
from gensim.models.phrases import Phraser, Phrases

all_words = [inaugural.words(x) for x in inaugural.fileids()]
phrases = Phrases(all_words,min_count=100,threshold=10)
bigram = Phraser(phrases)
print(bigram["Finest","People","in","United","States"])