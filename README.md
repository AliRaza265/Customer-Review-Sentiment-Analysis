# Customer Review Sentiment Analysis

A Machine Learning and Natural Language Processing (NLP) project that analyzes customer reviews and classifies them into **Positive, Neutral, or Negative** sentiment.

The project uses classical NLP techniques such as text preprocessing and TF-IDF vectorization, followed by multiple Machine Learning algorithms. **Random Forest Classifier** is used as the final model.

---

## 📌 Project Overview

Customer reviews contain valuable information about how users feel about a product or service. Manually analyzing thousands of reviews can be difficult and time-consuming.

This project automates sentiment analysis by taking a customer review as input and predicting its sentiment:

* 🟢 **Positive**
* 🟡 **Neutral**
* 🔴 **Negative**

The trained model is integrated into a **Streamlit web application** where users can enter a review and receive a sentiment prediction.

---

## 🎯 Project Objectives

* Analyze customer review text using NLP techniques.
* Clean and preprocess raw text data.
* Convert text into numerical features using TF-IDF.
* Handle class imbalance using SMOTE.
* Compare different Machine Learning algorithms.
* Select a suitable final model based on evaluation results.
* Save the trained model and TF-IDF vectorizer.
* Build a user-friendly Streamlit application for sentiment prediction.

---

## 📊 Dataset

The project uses the `dataset_IGWA_FB10.csv` dataset.

### Dataset Information

* **Original records:** 176,250
* **Features:** 11
* **Sentiment classes:** 3
* **Missing values:** None
* **Exact duplicate rows:** 0
* **Unique review texts:** 176,250

### Sentiment Distribution

| Sentiment |     Records |
| --------- | ----------: |
| Negative  |      88,120 |
| Positive  |      77,697 |
| Neutral   |      10,433 |
| **Total** | **176,250** |

Due to local hardware limitations, **3,000 records were sampled** for model development using a fixed random state of `42`.

---

## 🧹 NLP Preprocessing

The following preprocessing steps are applied to the review text:

1. Convert text to lowercase.
2. Expand contractions.
3. Remove URLs.
4. Remove HTML content.
5. Remove punctuation.
6. Convert emojis into text descriptions.
7. Replace underscores with spaces.
8. Tokenize the text.
9. Remove English stopwords.
10. Apply Snowball stemming.
11. Join processed tokens back into text.

### Example

```text
Original:
"I loved this app 😍! It's really easy to use."

After preprocessing:
"love app smil face heart eye realli easi use"
```

---

## 🔢 Feature Extraction

After preprocessing, the text is converted into numerical features using:

### TF-IDF

**TF-IDF (Term Frequency–Inverse Document Frequency)** is used to represent the importance of words within the reviews.

The trained TF-IDF vectorizer is saved as:

```text
Models/Vector.pkl
```

---

## ⚖️ Handling Class Imbalance

The dataset contains fewer Neutral reviews compared with Positive and Negative reviews.

To address the imbalance during training, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training data.

The test data was kept separate and was not oversampled.

---

## 🤖 Machine Learning Models

Multiple Machine Learning algorithms were evaluated:

* Random Forest Classifier
* K-Nearest Neighbors
* Decision Tree Classifier
* Linear Support Vector Classifier

### Model Comparison

| Model             |   Accuracy |
| ----------------- | ---------: |
| **Random Forest** | **71.17%** |
| LinearSVC         |     64.67% |
| Decision Tree     |     64.17% |
| KNN               |     26.67% |

Based on the current evaluation, **Random Forest Classifier** was selected as the final model.

> Accuracy is not the only important metric because the Neutral class is comparatively smaller. Macro F1-score and the confusion matrix should also be considered when evaluating future versions of the model.

---

## 🌲 Final Model

The final model is:

```text
RandomForestClassifier
```

Configuration:

```text
n_estimators = 100
random_state = 42
```

The trained model is saved as:

```text
Models/Model.pkl
```

The TF-IDF vectorizer is saved as:

```text
Models/Vector.pkl
```

---

## 🖥️ Streamlit Application

The project includes a Streamlit web application.

### Application Workflow

```text
User enters a review
        ↓
Text preprocessing
        ↓
TF-IDF Vectorization
        ↓
Random Forest Model
        ↓
Sentiment Prediction
        ↓
Positive / Neutral / Negative
```

The application provides a simple interface where users can enter a customer review and receive the predicted sentiment.

---

## 📁 Project Structure

```text
Customer-Review-Sentiment-Analysis/
│
├── app.py
├── main.py
├── Data_processing.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data_set/
│   └── dataset_IGWA_FB10.csv
│
├── Models/
│   ├── Model.pkl
│   └── Vector.pkl
│
└── project requirement document/
    └── Customer_Review_Sentiment_Analysis_Project_Requirement_Document.docx
```

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### NLP

* NLTK
* Contractions
* Emoji

### Machine Learning

* Scikit-learn
* Imbalanced-learn

### Model

* Random Forest Classifier

### Visualization / Application

* Streamlit

### Model Serialization

* Pickle

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/AliRaza265/Customer-Review-Sentiment-Analysis.git
```

Move into the project directory:

```bash
cd Customer-Review-Sentiment-Analysis
```

Create a virtual environment:

```bash
python -m venv env
```

Activate the environment on Windows:

```bash
env\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📈 Evaluation

The project evaluates the models using:

* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1-score
* Classification Report

Because the dataset is imbalanced, especially for the Neutral class, **macro-level evaluation metrics should also be considered** when comparing future models.

---

## 🔮 Future Improvements

Possible future improvements include:

* Improve Neutral sentiment classification.
* Experiment with Logistic Regression and LinearSVC tuning.
* Compare different TF-IDF configurations.
* Experiment with word and character n-grams.
* Try Word2Vec or other word embeddings.
* Increase the training dataset when better hardware is available.
* Experiment with Deep Learning approaches.
* Implement LSTM/GRU-based sentiment classification.
* Explore Transformer-based models such as BERT.
* Improve Streamlit UI and add confidence/probability information where appropriate.

---

## 💡 Learning Outcomes

Through this project, I practiced:

* Data cleaning and exploration
* NLP preprocessing
* Tokenization
* Stopword removal
* Stemming
* Emoji preprocessing
* TF-IDF feature extraction
* Handling imbalanced datasets
* SMOTE
* Supervised Machine Learning
* Model comparison
* Model evaluation
* Model serialization
* Streamlit deployment

---

## 👨‍💻 Author

**Ali Raza**

Machine Learning / Python Developer

GitHub: [AliRaza265](https://github.com/AliRaza265)

---

## ⭐ Project Status

**Completed ✅**

This project demonstrates a complete classical NLP and Machine Learning workflow, from raw customer reviews to a deployed sentiment prediction application.
