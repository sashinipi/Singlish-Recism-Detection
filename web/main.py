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
            name = data['name']
            return jsonify({"method": "POST", "value": name})
        if request.method == "GET":
            return jsonify({"method": "GET"})

    except Exception as e:
        flash(e)
        return render_template("about.html")

@app.route("/srd/webapk", methods=["GET", "POST"])
def web_apk():
    try:
        data = request.form
        content = data['content']
        if len(content)>5:
            dict = {'result': 'Racism'}
        else:
            dict = {'result': 'Neutral'}
        return render_template("result.html",result = dict)

    except Exception as e:
        flash(e)
        return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)