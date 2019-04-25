import os
from flask import Flask, request, abort, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from celery import Celery
from sqlalchemy import Column, Integer, String
from model_processing import proc


api = Flask(__name__)

api.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(api)

class Track(db.Model):

	id = db.Column(db.Integer, primary_key=True)
	name = db.Column(db.String(20), unique=False, nullable=False)
	status = db.Column(db.String(100), unique=False, nullable=False)
	
	def __repr__(self):
	    return f"Track('{self.name}')"

db.create_all()

UPLOAD_DIRECTORY = "files"
GENRES_DIRECTORY = "models"
DOWNLOAD_DIRECTORY = "proc_files"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

if not os.path.exists(DOWNLOAD_DIRECTORY):
    os.makedirs(DOWNLOAD_DIRECTORY)

api.config['CELERY_BROKER_URL'] = 'amqp://localhost//'
api.config['CELERY_RESULT_BACKEND'] = 'sqlite:///site.db'


# Initialize Celery
celery = Celery(api.name, broker=api.config['CELERY_BROKER_URL'])
celery.conf.update(api.config)


@celery.task
def process_midi(file, style, track_id):
    with api.app_context():
        track = Track.query.get(track_id)
        track.status = "Processing"
        db.session.add(track)
        db.session.commit()
        proc(file)
        track.status = "Ready"
        db.session.add(track)
        db.session.commit()


@api.route("/genres")
def list_genres():

    #Show all avalible genres
    genres = []
    for filename in os.listdir(GENRES_DIRECTORY):
        path = os.path.join(GENRES_DIRECTORY, filename)
        if os.path.isfile(path):
            if "DS" not in filename:
                genres.append(filename[:-3])
    return jsonify(genres)


@api.route("/proc_files/<path:path>")
def get_file(path):

    #Download a file
    return send_from_directory(DOWNLOAD_DIRECTORY, path, as_attachment=True)


@api.route("/files/<filename>", methods=["POST"])
def post_file(filename):
    
    #Upload a file
    if "/" in filename:
        # Return 400 BAD REQUEST
        abort(400, "no subdirectories directories allowed")

    with open(os.path.join(UPLOAD_DIRECTORY, filename), "wb") as fp:
        fp.write(request.data)

    style = request.args.get("style")
    track_id = request.args.get("id")
    status = "Waiting for process"

    track = Track(id=track_id, name=filename, status=status)
    print(track)
    db.session.add(track)
    db.session.commit()

    print("start_processing")
    process_midi("files/" + filename, style, track_id)

    return "", 201

@api.route("/status/<id>")
def get_status(id):
	track = Track.query.get(id)
	print(track)
	tracks = []
	tracks.append(track.status)
	return jsonify(tracks)

if __name__ == "__main__":
    api.run(debug=True, port=8000)

