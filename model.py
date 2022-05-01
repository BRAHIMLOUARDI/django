import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

path_to_data = './backend/fra_eng_sc.txt'


translation_file = open(path_to_data,"r", encoding='utf-8') 
raw_data = translation_file.read()
translation_file.close()


raw_data = raw_data.split('\n')
pairs = [sentence.split('\t') for sentence in  raw_data]
# mx=[pair[0] for pair in pairs]
# print(mx)

def clean_sentence(sentence):
    lower_case_sent = sentence.lower()
    string_punctuation = "?" + "¡" + '¿'+"."+"!"+'' 
    # string_punctuation =string.punctuation.replace("'",'').replace("-",'')+ "¡" + '¿'

    clean_sentence = lower_case_sent.translate(str.maketrans('', '', string_punctuation))
   
    return clean_sentence
def tokenize(sentences):
    # Create tokenizer
    text_tokenizer = Tokenizer()
    # Fit texts
    text_tokenizer.fit_on_texts(sentences)
    return text_tokenizer.texts_to_sequences(sentences), text_tokenizer
english_sentences = [clean_sentence(pair[0]) for pair in pairs]
french_sentences = [clean_sentence(pair[1]) for pair in pairs]


fra_text_tokenized, fra_text_tokenizer = tokenize(french_sentences)
eng_text_tokenized, eng_text_tokenizer = tokenize(english_sentences)





french_vocab = len(fra_text_tokenizer.word_index) + 1
# print(fra_text_tokenizer.word_index)
english_vocab = len(eng_text_tokenizer.word_index) + 1
# print("french vocabulary is of {} unique words".format(french_vocab))
# print("English vocabulary is of {} unique words".format(english_vocab))


max_french_len = int(len(max(fra_text_tokenized,key=len)))
max_english_len = int(len(max(eng_text_tokenized,key=len)))

fra_pad_sentence = pad_sequences(fra_text_tokenized,15, padding = "post")
eng_pad_sentence = pad_sequences(eng_text_tokenized,15, padding = "post")




fra_pad_sentence = fra_pad_sentence.reshape(*fra_pad_sentence.shape, 1)
eng_pad_sentence = eng_pad_sentence.reshape(*eng_pad_sentence.shape, 1)



def logits_to_sentence(model,sentence):

    x=[sentence]

    y=fra_text_tokenizer.texts_to_sequences(x)
    z=pad_sequences(y, 15, padding = "post")
    # print(z)
    z=np.reshape(z,(1,15,1))
    logits=model.predict(z)[0]
    index_to_words = {idx: word for word, idx in eng_text_tokenizer.word_index.items()}
    index_to_words[0] = '' 
    predict= ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])
  
    return  predict
           
