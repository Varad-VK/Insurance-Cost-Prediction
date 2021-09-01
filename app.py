import app_ml
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def root():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit():
    name = request.form.get('name')
    age = int(request.form.get('age'))
    gender = int(request.form.get('gender'))
    bmi = int(request.form.get('bmi'))
    children = int(request.form.get('children'))
    smoker = int(request.form.get('smoker'))
    region = int(request.form.get('region'))
    algorithm = int(request.form.get('algorithm'))

    print(request.form)

    charges = app_ml.predict(age, gender, bmi, children, smoker, region, algorithm)

    return render_template("result.html", charges=charges, name=name)
    # return str(charges)


app.run(host='0.0.0.0', port=4000, debug=True)
