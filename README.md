# Fake vs True News Classification and Summarization

This project classifies news articles as **True** or **Fake** using machine learning models and provides summaries using both traditional and transformer-based approaches.

---

### **Overview**
- **Classification**: Predicts whether a news article is **True** or **Fake** using the following models:
  - Logistic Regression
  - Decision Tree Classifier
  - Gradient Boosting Classifier
  - Random Forest Classifier
- **Summarization**: Generates a summary of the news using:
  - GloVe word embeddings with cosine similarity
  - BART transformer model (Facebook's BART-large-CNN)

---

### **Installation**
Install required dependencies:

```bash
pip install -r requirements.txt
```
Or individually:

```bash
pip install pandas numpy scikit-learn seaborn gensim nltk transformers
```

---

### **Data Preprocessing**
- Load `Fake.csv` and `True.csv` datasets.
- Clean the text (remove URLs, special characters, etc.).
- Split data into **train** and **test** sets (75% / 25%).

---

### **Model Training**
Four classifiers are trained:
1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Gradient Boosting Classifier**
4. **Random Forest Classifier**

Models are evaluated using **accuracy** and **classification report**.

---

### **Manual Testing**
Test the models by entering news text. The models classify the news and display the results:

```python
def manual_testing(news):
    ...
    print("LR: {} DT: {} GBC: {} RFC: {}".format(...))
```

---

### **Summarization**
- **Traditional**: GloVe embeddings + cosine similarity to summarize text.
- **Transformer**: **BART** (Bidirectional and Auto-Regressive Transformers) for advanced summarization.

Example:

```python
def summarize_sentence_bart(sentence):
    summary = summarizer(sentence)
    return summary[0]['summary_text']
```

---

### **Save and Load Models**
Save models and vectorizer using `joblib`:

```python
import joblib
joblib.dump(LR, 'lr_model.pkl')
```

Load them later:

```python
LR = joblib.load('lr_model.pkl')
```

---

### **Usage**
1. **Classify News**: Classify text as **True** or **Fake**.
   ```python
   news = input("Enter news: ")
   manual_testing(news)
   ```
2. **Summarize News**: Get a summary of the news.
   ```python
   summary = summarize_sentence_bart(input_sentence)
   ```

---

### **Dependencies**
- `pandas`, `numpy`, `scikit-learn`, `seaborn`, `gensim`, `nltk`, `transformers`

---

### **Conclusion**
This project provides a solution for classifying news articles and generating summaries using both traditional ML and advanced transformer models.
