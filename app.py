import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
import streamlit as st



# Começamos o desenvolvimento do nosso app
st.title("Processamento de textos com Machine Learning.")

# aqui definimos alguns dos elementos do front end da página da web, como
# a fonte e a cor de fundo, o preenchimento e o texto a ser exibido
html_temp = """
<div style ="background-color:blue;padding:13px">
<h1 style ="color:white;text-align:center;">Streamlit NLP Classifier APP </h1>
</div>
"""

# esta linha nos permite exibir os aspectos do front end que temos
# definido no código acima
st.markdown(html_temp, unsafe_allow_html = True)

# definindo a função que fará a previsão usando os dados que o usuário insere
def prediction(docs_new):

    # carregar o modelo treinado
    model = joblib.load('mlp_model.pkl')

    docs_new = str(docs_new)
    vectorizer = TfidfVectorizer()   
    X_new_tfidf_vectorize = vectorizer.transform(docs_new)

    sgd_predicted = model.predict(X_new_tfidf_vectorize)

    categories = ['rec.motorcycles', 'rec.autos']
    twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)

    for doc, category in zip(docs_new, sgd_predicted):
        return 'O texto a seguir..: {}, foi classificado como {}. '.format(doc, twenty_train.target_names[category]) 

# esta é a função principal com a qual definimos nossa página da web
def main():

    # Opções de predição
    st.subheader('** Selecione uma das opções abaixo:**')
    options = st.radio('O que deseja fazer?', ('Classificação de textos', 'Geração de textos', 'Predição em batch'))

    if options == 'Classificação de textos':
        # as linhas a seguir criam caixas de texto nas quais o usuário pode inserir
        # os dados necessários para fazer a previsão
        text=st.text_input("Digite aqui seu texto aqui:")
        result =" "
        # a linha abaixo garante que quando o botão chamado Predict for clicado,
        # a função de predict definida acima é chamada para fazer a previsão
        # e armazene-o no resultado variável
        if st.button("Predict"):
            result = prediction(text)
        st.success('The output is {}'.format(result))
    if options == 'Geração de textos':

        text, qtd = st.text_input("Digite um tema aqui:"), st.text_input("Quantidade de palavras a serem geradas:")
        

        st.write('Tema para próxima aula')


    if options == 'Predição em batch':

        st.write('Selecione um arquivo csv ou xlsx para upload.')
        st.file_uploader('load File')

        st.write('Tema para próxima aula')

if __name__=='__main__':
    main()