from flask import Flask, request
from pipeline import FullPipeline
from models import Models

app = Flask(__name__)


@app.route("/", methods=["POST"])
def home():
    response = []
    if all([len(request.form["title"]) > 0, len(request.form["abstract"]) > 0]):
        pipeline = FullPipeline(request.form["title"], request.form["abstract"])
        pipeline.run()

        models = Models()
        models.load()
        for name, model in models.models.items():
            if model.predict(pipeline.data) == 1:
                response.append(models.labels_dict[name])

    return response
