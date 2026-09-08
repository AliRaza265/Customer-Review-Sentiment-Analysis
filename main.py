import pandas as pd 
from contractions import fix
import re 
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk 
import emoji
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle as pkl
# download sum additional data 
nltk.download("punkt")
nltk.download("stopwords")


stop_word = set(stopwords.words("english"))
stemer  = SnowballStemmer("english")

# Data Anylsis with pandas  
read_csv = pd.read_csv("data_set/dataset_IGWA_FB10.csv", usecols = ["original_text","sentiment_text"])
read_csv = read_csv.sample(n  = 3000 ,random_state = 42)
print(read_csv.head)
print(read_csv.info())
print(read_csv.describe())

# filtering Data 
find_missing_values = read_csv.isnull().sum()
print(find_missing_values)
find_duplicate_values = read_csv.duplicated().sum()
print(find_duplicate_values)




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

# sample data for testing the fuction 
sample_string = r"I love it when professors draw a big question mark next to my answer on an exam because I’m always like yeah I don’t either ¯\_(ツ)_/¯ @VolphanCarol @littlewhitty @mysticalmanatee https://t.co/yZlafy0lsd <h1>hello</h1> 🤪 growthing referring teachings"
sample_test = data_processing(sample_string)  
print(sample_test)

# data_processing on label column
read_csv["Prs_data"] = read_csv["original_text"].apply(lambda x : data_processing(x))
print(read_csv["Prs_data"].sample(10))


# apply maping techinquie on sentiment_text
read_csv["sentiment_text"] = read_csv["sentiment_text"].map({"Negative" : 0 , "Positive" : 1 , "Neutral" : 2}) 
print(read_csv["sentiment_text"].sample(10))

# split data for traing and testing 
X = read_csv[["Prs_data"]]
y = read_csv["sentiment_text"]
x_train,x_test,y_train,y_test = train_test_split(X,y,random_state = 42 ,test_size = 0.2)
print(x_test)

# Apply Tfidf Vectorizer 
vector =  TfidfVectorizer()
x_train_vector = vector.fit_transform(x_train["Prs_data"])
x_test_vector = vector.transform(x_test["Prs_data"])
print(x_test_vector.toarray())

# Balance data by using  SMOTE technique 
smote_sampling = SMOTE(random_state=42)
x_train_sample,y_train_sample = smote_sampling.fit_resample(x_train_vector,y_train)
print(y_train_sample.value_counts())

# Apply RandomForestClassifier for Training  Prediction 
rfc_model = RandomForestClassifier(random_state = 42)
rfc_model.fit(x_train_sample,y_train_sample)
rfc_pred = rfc_model.predict(x_test_vector)

# Evaluate Model Perfomance 
print(f"rfc_model Accuracy : {accuracy_score(rfc_pred,y_test)}")
print(f"rfc_model Confusion_Matrix : {confusion_matrix(rfc_pred,y_test)}")
print(f"rfc_model Classification_Report : {classification_report(rfc_pred,y_test)}")

# Apply KNeighborsClassifier for Training  Prediction 
knc_model = KNeighborsClassifier()
knc_model.fit(x_train_sample,y_train_sample)
knc_pred = knc_model.predict(x_test_vector)

# Evaluate Model Perfomance 
print(f"knc_model Accuracy : {accuracy_score(knc_pred,y_test)}")
print(f"knc_model Confusion_Matrix : {confusion_matrix(knc_pred,y_test)}")
print(f"knc_model Classification_Report : {classification_report(knc_pred,y_test)}")

# Apply Model for Training  Prediction 
dtc_model = DecisionTreeClassifier()
dtc_model.fit(x_train_sample,y_train_sample)
dtc_pred = dtc_model.predict(x_test_vector)

# Evaluate Model Perfomance 
print(f"dtc_model Accuracy : {accuracy_score(dtc_pred,y_test)}")
print(f"dtc_model Confusion_Matrix : {confusion_matrix(dtc_pred,y_test)}")
print(f"dtc_model Classification_Report : {classification_report(dtc_pred,y_test)}")

# Apply LinearSVC for Training  Prediction 
ls_model = LinearSVC()
ls_model.fit(x_train_sample,y_train_sample)
ls_pred = ls_model.predict(x_test_vector)

# Evaluate Model Perfomance 
print(f"ls_model Accuracy : {accuracy_score(ls_pred,y_test)}")
print(f"ls_model Confusion_Matrix : {confusion_matrix(ls_pred,y_test)}")
print(f"ls_model Classification_Report : {classification_report(ls_pred,y_test)}")

# save model for use app.py 
pkl.dump(rfc_model,open("Models/Model.pkl","wb"))
pkl.dump(vector,open("Models/Vector.pkl","wb"))