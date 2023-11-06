from flask import Flask, request, jsonify, render_template
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
import string
nltk.download("stopwords")
nltk.download('wordnet')
import re
nltk.download('averaged_perceptron_tagger')
import pickle
import string
import numpy as np
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def hello_world():
    return render_template("index.html")
@app.route('/predict',methods=['POST'])
def predict():
    text=request.form.get("inputxt")
    print(text)
    text=preprocess_and_lemmenizate(text)
    prediction = model.predict([text])[0]
    predict_emotion=predict_label(prediction)
    return render_template('index.html', prediction_text='Predicted emotion : {}'.format(predict_emotion))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    text=data["text"]
    text=preprocess_and_lemmenizate(text)
    prediction = model.predict([text])
    output = prediction[0]
    out_dict={"prediction":predict_label(output)}
    # print(out_dict)
    return jsonify(out_dict)


def preprocess_and_lemmenizate(text):
    punctuations=string.punctuation
    text=text.translate(str.maketrans("","",punctuations))
    stopWords=set(stopwords.words("english"))
    text=" ".join([word for word in text.split() if word not in stopWords])
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    text = re.sub('\s+', ' ', text)
    lemmatizer = WordNetLemmatizer()
    wordnet_map = {"N":wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}
    pos_text = pos_tag(text.split())
    text=" ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_text])
    return text



def predict_label(prediction):
    if(prediction==0):
        return "sadness"
    elif(prediction==1):
        return "anger"
    elif(prediction==2):
        return "love"
    elif(prediction==3):
        return "surprise"
    elif(prediction==4):
        return "fear"
    else:
        return "joy"

if __name__ == '__main__':
    app.run(debug=True)








    
















