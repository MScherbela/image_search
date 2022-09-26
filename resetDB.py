from app import app
from extensions import db
import os
from flask_security.utils import hash_password
import datetime

if os.path.isfile('/data/image_search.db'):
    os.remove('/data/image_search.db')

with app.app_context():
    user_datastore = app.extensions['security'].datastore
    db.create_all()
    user_datastore.create_role(name='admin')
    db.session.commit()

    user = user_datastore.create_user(email="michael.scherbela@gmail.com",
                               password=hash_password(os.environ.get("ADMIN_PASSWORD", "password")),
                               roles=['admin'])
    user.confirmed_at = datetime.datetime.now()
    db.session.commit()



