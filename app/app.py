# !/usr/bin/env python
# title           :app.py
# description     :Flask App
# author          :Sebastian Maldonado
# date            :7/22/2020
# version         :0.0
# usage           :SEE README.md
# notes           :Enter Notes Here
# python_version  :3.7.7
# conda_version   :4.8.2
# tf_version      :1.14
# =================================================================================================================

from flask import Flask, render_template, url_for, request, redirect, config
import scripts

app = Flask(__name__)


@app.route('/')
@app.route('/home')  # Root page of a website
def home():
    """
    :return: Render Home Page
    """
    return render_template('home.html')


@app.route('/classify_news', methods=['GET', 'POST'])
def submit_article():
    """
    Processes inference for user's news article
    :return: Inference result
    """
    if request.method == "POST":
        # Capture Article URL
        url = request.form.get('url_page')
        article = scripts.scrape_article(url)
        text = scripts.clean_text(article[3])

        prediction = scripts.perform_inference(text)

        return render_template('results.html', prediction=prediction[0][0], title=article[0], text=article[1],
                               img=article[2])

    return render_template('classify_news.html', title='Upload')


if __name__ == '__main__':
    app.run(debug=1)  # Debug mode activated
