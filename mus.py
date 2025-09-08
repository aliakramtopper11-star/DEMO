from flask import Flask, request, render_template_string
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import joblib
import numpy as np
import os

app = Flask(__name__)

MODEL_FILE = "knn_iris_model.joblib"

# Step 1: Train and save the model if not exists
if not os.path.exists(MODEL_FILE):
    iris = load_iris()
    X, y = iris.data, iris.target
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
else:
    model = joblib.load(MODEL_FILE)

# Step 2: Define class names
iris = load_iris()
class_names = iris.target_names

# Step 3: HTML template with form
HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <title>Iris KNN Prediction</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 40px; }
    label { display: block; margin-top: 10px; }
    input[type=number] { width: 100px; }
    button { margin-top: 20px; padding: 10px 15px; }
    .result { margin-top: 20px; font-weight: bold; }
  </style>
</head>
<body>
  <h2>Iris Flower Class Prediction using KNN</h2>
  <form method="POST" action="/">
    <label>Sepal Length:
      <input type="number" name="sepal_length" step="any" required>
    </label>
    <label>Sepal Width:
      <input type="number" name="sepal_width" step="any" required>
    </label>
    <label>Petal Length:
      <input type="number" name="petal_length" step="any" required>
    </label>
    <label>Petal Width:
      <input type="number" name="petal_width" step="any" required>
    </label>
    <button type="submit">Predict</button>
  </form>
  {% if prediction %}
    <div class="result">
      Predicted Iris Class: <span style="color:green;">{{ prediction }}</span>
    </div>
  {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            # Extract features from form and convert to float
            features = [
                float(request.form['sepal_length']),
                float(request.form['sepal_width']),
                float(request.form['petal_length']),
                float(request.form['petal_width'])
            ]
            features_np = np.array(features).reshape(1, -1)
            pred_class_idx = model.predict(features_np)[0]
            prediction = class_names[pred_class_idx]
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template_string(HTML_PAGE, prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
