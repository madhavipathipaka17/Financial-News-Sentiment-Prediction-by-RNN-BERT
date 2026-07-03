# 📈 Financial News Sentiment Prediction using Deep Learning & BERT

A Natural Language Processing (NLP) project that classifies
finance-related tweets into **Bearish**, **Bullish**, and **Neutral**
sentiments using Deep Learning (Simple RNN, LSTM, GRU) and a fine-tuned
**BERT** model.

## 🚀 Features

-   Financial tweet sentiment classification
-   Deep Learning baseline models (RNN, LSTM, GRU)
-   Fine-tuned BERT model
-   Interactive Streamlit dashboard
-   Performance comparison using Accuracy and Macro F1-score

## 🛠️ Technologies

-   Python
-   PyTorch
-   Hugging Face Transformers
-   Hugging Face Datasets
-   Scikit-learn
-   Pandas
-   NumPy
-   Streamlit

## 📂 Dataset

  Item                                       Value
  -------------------- ---------------------------
  Training Samples                           9,543
  Validation Samples                         2,388
  Classes                Bearish, Bullish, Neutral

Labels: - 0 → Bearish - 1 → Bullish - 2 → Neutral

## 🧹 Text Preprocessing

-   Lowercasing
-   URL removal
-   Mention removal
-   Extra whitespace removal
-   Preserve stock tickers (e.g. `$AAPL`)
-   Tokenization
-   Sequence padding (RNN models)

> Note: BERT uses the original text with its own tokenizer.

## 🤖 Models

  Model            Accuracy    Macro F1
  ------------ ------------ -----------
  Simple RNN         65.62%        0.27
  LSTM               73.62%        0.56
  GRU                80.74%        0.72
  **BERT**       **88.44%**   **0.847**

## 🏆 Best Model

Fine-tuned **BERT** achieved the highest performance due to its
transformer architecture and bidirectional contextual understanding.

## 💻 Streamlit Dashboard

The application allows users to:

-   Enter a financial tweet or headline
-   Predict Bearish, Bullish, or Neutral sentiment
-   View prediction probabilities
-   Visualize class probabilities with a bar chart
-   View validation class distribution

## 📁 Project Structure

``` text
Financial-News-Sentiment/
├── data/
├── notebooks/
├── finbert_model/
├── streamlit_app/
│   └── app.py
├── requirements.txt
├── README.md
└── comparison_report.pdf
```

## ⚙️ Installation

``` bash
git clone https://github.com/yourusername/Financial-News-Sentiment.git
cd Financial-News-Sentiment
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Evaluation Metrics

-   Accuracy
-   Precision
-   Recall
-   Macro F1-score
-   Confusion Matrix

## 🔮 Future Improvements

-   FinBERT
-   RoBERTa
-   Live financial news API
-   Docker deployment
-   Cloud deployment

## 👩‍💻 Author

**Madhavi Pathipaka**
