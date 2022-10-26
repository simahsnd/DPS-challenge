from flask import Flask, request, render_template
from Predictor import predict

app = Flask(__name__)

@app.route('/')
def landing_page():
    return render_template('landing_page.html')

@app.route('/predictor/')
def predictor():
    predicted_value = predict()
    return render_template('pipeline_predictor.html', predicted_value=predicted_value)

if __name__=='__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)