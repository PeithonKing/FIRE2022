from flask import Flask, render_template

app = Flask(__name__)

@app.route('/run')
def hello_world():
    # return "Hello World!"
	return render_template('index.html',
                        tweet_id="1234567890123456789",
                        tweet="This is a tweet",
                        )

app.run(host="localhost", port=5000, debug=True)