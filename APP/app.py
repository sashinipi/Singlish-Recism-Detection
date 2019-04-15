from flask import Flask, render_template, request, flash, jsonify
from main.srd import SRD
from main.multinomial_db import MultinomialNBC
from main.svm import SVM

app = Flask(__name__)

@app.route("/")
def default():
    return "Flask app running..."

@app.route("/srd/home")
def home():
    return render_template("home.html")

@app.route("/srd/about")
def about():
    return render_template("about.html")

@app.route("/srd/apk", methods=["GET", "POST"])
def apk():
    try:
        if request.method == "POST":
            return jsonify(srd_obj.predict(request.json))
        if request.method == "GET":
            return jsonify({"Status":"Up", "method": "GET"})

    except Exception as e:
        print(e)
        return render_template("about.html")

@app.route("/srd/webapk", methods=["GET", "POST"])
def web_apk():
    try:
        return render_template("result.html", result = srd_obj.predict(request.form))
    except Exception as e:
        flash(e)
        return render_template("about.html")

if __name__ == "__main__":
    srd_obj = SRD()
    print("Test:", srd_obj.predict( {"content":"This is test", "type":'1'}))
    # app.run(debug=True, host = '0.0.0.0', port=5000) # use this for docker
    app.run(debug=True, host = '127.0.0.1', port=5000)