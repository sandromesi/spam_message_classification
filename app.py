from flask import Flask, request, render_template
import pickle
import pandas as pd

#https://unicode.org/emoji/charts-14.0/full-emoji-list.html

with open('email_classification.pickle', 'rb') as f:
        model = pickle.load(f)
df = pd.read_csv('spam.csv')

def classify_message(text):
    return model.predict([text])[0]

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():

    if request.method == 'POST':
        try:
            email_body = request.form['email_body']
        except:
            return render_template('index.html', 
            fill_message='Please, message is required!')

        if email_body == '':
            return render_template('index.html', 
            fill_message='Please, message is required!')
        
        prediction = classify_message(email_body)

        if prediction == 'ham':
            prediction = 'This message is not spam!' + '\N{grinning face}'

        if prediction == 'spam':
            prediction = 'Watch out! This message may be Spam!' + '\N{fearful face}'

        return render_template('prediction.html', 
        prediction=prediction, email_body=email_body)

if __name__ == '__main__':

    app.run()