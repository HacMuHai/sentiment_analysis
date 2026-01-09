import pandas as pd
import streamlit as st
from PIL import Image
from models import NaiveBayesModel, SVMModel, LSTMModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(page_title='Sentiment Analysis', page_icon='😃', layout='wide', initial_sidebar_state='auto')

# Load the dataset
@st.cache_data
def load_data():
    csv_file_path = './data/train.csv'
    df = pd.read_csv(csv_file_path)
    df = df.drop(columns=['selected_text'])
    df["text"] = df["text"].astype(str)
    return train_test_split(df, test_size=0.2, random_state=42)

train_df, test_df = load_data()

def main():
    # Add a banner image
    banner_image = Image.open("./images/sentimentanalysishotelgeneric-2048x803-1.jpg")
    st.image(banner_image, use_column_width=True)

    st.title("So sánh các mô hình phân loại cảm xúc")

    # Train all models
    @st.cache_resource(show_spinner=True)
    def train_all_models():
        nb = NaiveBayesModel(); nb.train(train_df)
        svm = SVMModel(); svm.train(train_df)
        lstm = LSTMModel(); lstm.train(train_df)
        return nb, svm, lstm
    nb_model, svm_model, lstm_model = train_all_models()

    # Evaluate models and cache results
    @st.cache_data(show_spinner=True)
    def evaluate_models(_nb, _svm, _lstm):
        accuracy_nb = _nb.evaluate(test_df)
        accuracy_svm = _svm.evaluate(test_df)
        accuracy_lstm = _lstm.evaluate(test_df)
        return accuracy_nb, accuracy_svm, accuracy_lstm
    accuracy_nb, accuracy_svm, accuracy_lstm = evaluate_models(nb_model, svm_model, lstm_model)

    # Display accuracy results
    st.subheader('Độ chính xác của các mô hình')
    accuracy_data = {
        'Model': ['Naive Bayes', 'SVM', 'LSTM'],
        'Accuracy (%)': [accuracy_nb * 100, accuracy_svm * 100, accuracy_lstm * 100]
    }
    accuracy_df = pd.DataFrame(accuracy_data)
    st.table(accuracy_df)

    # Create accuracy comparison chart
    fig = px.bar(accuracy_df, x='Model', y='Accuracy (%)', text='Accuracy (%)')
    st.plotly_chart(fig)

    # Prediction UI
    st.subheader('Dự đoán cảm xúc với từng mô hình')
    user_input = st.text_area("Nhập đoạn văn bản:", "I love this product!", key="text_input")
    if st.button("Phân loại cảm xúc", key="analyze_button"):
        pred_nb, _ = nb_model.predict(user_input)
        pred_svm, _ = svm_model.predict(user_input)
        pred_lstm, _ = lstm_model.predict(user_input)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Naive Bayes:** {pred_nb}")
        with col2:
            st.write(f"**SVM:** {pred_svm}")
        with col3:
            st.write(f"**LSTM:** {pred_lstm}")

if __name__ == "__main__":
    main()
