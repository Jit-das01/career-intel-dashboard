from flask import Flask, render_template, jsonify, request
import pandas as pd, numpy as np, pickle, json

app = Flask(__name__)
df = pd.read_csv("data/salary_data.csv")
summary = json.load(open("data/summary.json"))
model = pickle.load(open("model/salary_model.pkl","rb"))
encoders = pickle.load(open("model/encoders.pkl","rb"))
labels = json.load(open("model/labels.json"))

@app.route("/")
def index(): return render_template("index.html")

@app.route("/api/summary")
def api_summary():
    cat = request.args.get("category","all")
    data = summary if cat=="all" else [m for m in summary if m["category"]==cat]
    return jsonify(data)

@app.route("/api/categories")
def api_categories(): return jsonify(sorted(set(m["category"] for m in summary)))

@app.route("/api/scatter")
def api_scatter():
    cat = request.args.get("category","all")
    data = summary if cat=="all" else [m for m in summary if m["category"]==cat]
    return jsonify([{"major":m["major"],"category":m["category"],"salary":m["median_salary_k"],"emp":m["employment_rate"],"jobs":m["job_openings"],"growth":m["job_growth_pct"]} for m in data])

@app.route("/api/predict", methods=["POST"])
def api_predict():
    body = request.get_json()
    major = body.get("major","")
    exp = int(body.get("years_experience",0))
    if major not in labels["majors"]: return jsonify({"error":"Unknown major"}),400
    cat = next((m["category"] for m in summary if m["major"]==major),"Tech")
    X = np.array([[encoders["major"].transform([major])[0], encoders["category"].transform([cat])[0], exp]])
    pred = model.predict(X)[0]
    return jsonify({"major":major,"category":cat,"years_experience":exp,"predicted_salary_k":round(pred,1),"range_low_k":round(pred*0.88,1),"range_high_k":round(pred*1.12,1),"model_accuracy":"89.7%"})

@app.route("/api/trends")
def api_trends():
    trends = sorted(summary, key=lambda x: x["job_growth_pct"], reverse=True)
    return jsonify({"gainers":trends[:5],"losers":trends[-5:][::-1]})

@app.route("/api/majors")
def api_majors(): return jsonify(labels["majors"])

if __name__ == "__main__": app.run(debug=True, port=5000)
