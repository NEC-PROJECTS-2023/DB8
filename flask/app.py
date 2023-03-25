from flask import Flask,render_template,request
import tensorflow as tf
import re
import string
from nltk.corpus import stopwords
from wordcloud import WordCloud,STOPWORDS
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
import numpy as np
 
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    data = request.form['data']
    print(data)
    model_json = open('./models/model1.json', 'r')
    loaded_model_json = model_json.read()
    model_json.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights("./models/model1.h5")
    #model = tf.keras.models.load_model('./models/fullmodel.h5',compile=False)
    def review_cleaning(text):
        '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
        and remove words containing numbers.'''
        text = str(text).lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        return text
    text = review_cleaning(data)
    stop_words = set(stopwords.words("english"))
    #Performing stemming on the review dataframe
    ps = PorterStemmer()
    news = re.sub('[^a-zA-Z]', ' ', text)
    news= news.lower()
    news = news.split()
    news = [ps.stem(word) for word in news if not word in stop_words]
    news = ' '.join(news)
    onehot_repr = one_hot(news,10000)
    embedded_docs=pad_sequences([onehot_repr],padding='pre',maxlen=5000)
    y = model.predict(embedded_docs)
    if y > 0.89:
        data = True
    else:
        data = False

    

    return render_template('classify.html', news = news, classify=data)

app.run(port=5000)