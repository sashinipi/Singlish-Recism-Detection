from flask import Flask, render_template, request, flash, jsonify
from main.simple_nn import SimpleNN

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
            return jsonify({"Status":"Up", "content": name, "prediction": "Racist", "confidence": 0.85})
        if request.method == "GET":
            return jsonify({"Status":"Up", "method": "GET"})

    except Exception as e:
        flash(e)
        return render_template("about.html")

@app.route("/srd/webapk", methods=["GET", "POST"])
def web_apk():
    try:
        data = request.form
        content = str(data['content'])
        type = str(data['type'])
        if type == 'simpleNN':
            p_class, conf = snn.predict_api(content)
        else:
            p_class, conf = "--", "--"
        dict = {'Prediction': p_class, 'Confidence': conf, 'Content': content, 'Type': type}

        return render_template("result.html", result = dict)

    except Exception as e:
        flash(e)
        return render_template("about.html")

if __name__ == "__main__":
    snn = SimpleNN()
    snn.load_values()

    print("Test:", snn.predict_api("This is test"))
    app.run(debug=True)