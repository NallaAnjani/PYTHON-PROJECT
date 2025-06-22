# from flask import Flask, request, render_template
# import joblib

# app = Flask(__name__)

# # Load trained models and vectorizer
# rf_model = joblib.load("rf_model.pkl")
# ab_model = joblib.load("ab_model.pkl")
# vectorizer = joblib.load("vectorizer.pkl")

# @app.route("/", methods=["GET", "POST"])
# def index():
#     result_rf = ""
#     result_ab = ""

#     if request.method == "POST":
#         email = request.form["email"]
#         vect_email = vectorizer.transform([email])

#         pred_rf = rf_model.predict(vect_email)[0]
#         result_rf = "SPAM (Bagging)" if pred_rf == 1 else "NOT SPAM (Bagging)"

#         pred_ab = ab_model.predict(vect_email)[0]
#         result_ab = "SPAM (Boosting)" if pred_ab == 1 else "NOT SPAM (Boosting)"

#     return render_template("form.html", result_rf=result_rf, result_ab=result_ab)

# if __name__ == "__main__":
#     app.run(debug=True)
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load models and vectorizer
rf_model = joblib.load("rf_model.pkl")  # Bagging
ab_model = joblib.load("ab_model.pkl")  # Boosting
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""

    if request.method == "POST":
        email = request.form["email"]
        vect_email = vectorizer.transform([email])

        # Get predictions
        pred_rf = rf_model.predict(vect_email)[0]
        pred_ab = ab_model.predict(vect_email)[0]

        # Final decision: if either model predicts spam, classify as spam
        if pred_rf == 1 or pred_ab == 1:
            result = "SPAM"
        else:
            result = "NOT SPAM"

    return render_template("form.html", result=result)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

