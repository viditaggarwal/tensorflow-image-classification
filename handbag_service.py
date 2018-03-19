from flask import request, url_for
# from flask.ext.api import FlaskAPI, status, exceptions
from flask import Flask, render_template
import json, time
from flask_cors import CORS
from handbag_model import test_image_url
import json

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)
CORS(app)

@app.route("/handbag", methods=["POST"])
def getResult(argLimit = None):
	data = json.loads(request.data)

	url = data['url']
	model_name = data['model']
	# url = str(request.form.get('url'))
	# model_name = str(request.form.get('model'))
	return json.dumps(test_image_url(model_name, url))


if __name__ == "__main__":
    app.run(host= '0.0.0.0')