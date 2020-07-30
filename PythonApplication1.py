#import basic library
import re
import string
import nltk 
#import for showing graphic
import matplotlib.pyplot as plt
#nltk.download()
# import StemmerFactory class
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# impor word_tokenize dari modul nltk
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# stemming process
sentence = "Perekonomian Indonesia sedang. dalam pertumbuhan yang yang yang membanggakan?"
cased_folding = sentence.strip().translate(str.maketrans("","",string.punctuation))
sentence_lower_cased = cased_folding.lower()
sentence_no_number = re.sub(r"\d+", "", sentence_lower_cased)
final_sentence = sentence_no_number
stemmed_sentence = stemmer.stem(final_sentence)
tokenized_sentences = nltk.tokenize.word_tokenize(stemmed_sentence)
kemunculan = nltk.FreqDist(tokenized_sentences)
print(kemunculan.most_common())


print("original sentence")
print(sentence)
print("steemed sentence")
print(stemmed_sentence)
print('tokenized sentence')
print(tokenized_sentences)

##Showing Graphic kemunculuan
kemunculan.plot(30,cumulative=False)
plt.show()

