import flask
from extensions import scheduler, init_extensions, photo_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField, IntegerField
from wtforms.validators import NumberRange
from flask_security.decorators import login_required
import PIL.Image, PIL.ImageOps
import uuid
from index_images import FeatureCalculator, load_data, index_directory
import numpy as np
import os
import os.path
import exif
import datetime

PHOTO_DIR = os.environ["PHOTO_DIR"]
UPLOAD_DIR = os.environ["UPLOAD_DIR"]
THUMBNAIL_DIR = os.environ["THUMBNAIL_DIR"]

def create_thumbnail(fname, width=512):
    img = PIL.Image.open(fname)
    img = img.convert("RGB")
    img = PIL.ImageOps.exif_transpose(img)
    scaling_factor = width / img.size[0]
    img = img.resize([int(x * scaling_factor) for x in img.size])
    thumbnail_fname = THUMBNAIL_DIR + str(uuid.uuid4()) +".jpg"
    img.save(thumbnail_fname)
    return thumbnail_fname


app = flask.Flask(__name__)
app.config.from_pyfile('config.py')
init_extensions(app)
feature_calc = FeatureCalculator(jit_fwd=False)
photo_ids, photo_features, photo_fnames = load_data()

def find_closest_images(reference_img_fname, n=48):
    x = feature_calc.process_image(reference_img_fname)
    dist = np.linalg.norm(x - photo_features, axis=1)
    ind_closest = np.argsort(dist)[:n]
    return [[photo_ids[i], dist[i], get_metadata(photo_fnames[photo_ids[i]])] for i in ind_closest]

def get_metadata(fname):
    try:
        with open(fname, 'rb') as f:
            img = exif.Image(f)
        data = dict(fname=fname,
                    model=img.get("model", ""),
                    date=img.get("datetime_original"))
    except Exception as e:
        data = dict(fname="", model="", date="")

    if data['date'] != "":
        date = datetime.datetime.strptime(data['date'], "%Y:%m:%d %H:%M:%S")
        data['date'] = "{:%d.%m.%Y, %H:%M}".format(date)
    return data

class AddDataForm(FlaskForm):
    photo = FileField("Photo to upload", validators=[FileAllowed(photo_uploads, 'Image only!'), FileRequired('File was empty!')])
    n_similar = IntegerField("Number of photos", validators=[NumberRange(min=1, max=100)], default=6)
    submit = SubmitField("Search for similar photos")


# %% Scheduler tasks
@scheduler.task(trigger='interval', hours=24)
def periodic_task():
    with app.app_context():
        index_directory(PHOTO_DIR)


# %% Views
@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    form = AddDataForm()
    if form.validate_on_submit():
        uploaded_img_fname = photo_uploads.save(form.photo.data)
        n_similar = int(form.n_similar.data)
        closest_images = find_closest_images(os.path.join(UPLOAD_DIR, uploaded_img_fname), n_similar)
    else:
        uploaded_img_fname = None
        closest_images = []
    return flask.render_template('index.html', form=form, uploaded_img=uploaded_img_fname, closest_images=closest_images)

@app.route('/stored_img/<id>')
@login_required
def stored_img(id):
    return flask.send_file(photo_fnames[id])

@app.route('/thumbnail/<id>')
@login_required
def thumbnail(id):
    thumbnail_fname = create_thumbnail(photo_fnames[id])
    return flask.send_file(thumbnail_fname)

@app.route('/uploaded_img/<fname>')
@login_required
def uploaded_img(fname):
    return flask.send_file(UPLOAD_DIR + fname)