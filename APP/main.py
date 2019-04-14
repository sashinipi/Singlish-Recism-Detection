from flask import Flask, render_template, request, flash, jsonify

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
            data = request.get_json()
            name = data['content']
            return jsonify({"Status":"Up", "content": name})
        if request.method == "GET":
            return jsonify({"Status":"Up", "method": "GET"})

    except Exception as e:
        flash(e)
        return render_template("about.html")

@app.route("/srd/webapk", methods=["GET", "POST"])
def web_apk():
    try:
        data = request.form
        content = data['content']
        if len(content)>5:
            dict = {'Prediction': 'Racism', 'Confidence': 0.5}
        else:
            dict = {'Prediction': 'Neutral', 'Confidence': 0.4}
        return render_template("result.html", result = dict)

    except Exception as e:
        flash(e)
        return render_template("about.html")


if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.run(debug=True)