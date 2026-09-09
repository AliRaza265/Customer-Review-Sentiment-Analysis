from contractions import fix
import re 
import emoji
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


stop_word = set(stopwords.words("english"))
stemer  = SnowballStemmer("english")



def data_processing(text):
    # convert string into lowercase
    text = text.lower()
    # fix constractions
    text = fix(text)
    # remove links with regular expression 
    text = re.sub(r"http\S+","",text)
    # remove html tags 
    text = re.sub(r"<.*?>","",text)
    # remove special characters 
    text = text.translate(str.maketrans("","",string.punctuation))
    # converting emojis into text 
    text = emoji.demojize(text,delimiters= (" " , " "))
    text = text.replace("_"," ")
    # split string into tokens 
    text = word_tokenize(text)
    # remove stop word 
    text = [word for word in text if word not in stop_word]
    # stem the data 
    text  = [stemer.stem(word) for word in text]
    # converting the word in to string rby using join
    text = " ".join(text)
    return (text)