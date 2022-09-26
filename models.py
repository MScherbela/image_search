from extensions import db
from datetime import datetime
from flask_security import UserMixin, RoleMixin

# Define models
roles_users = db.Table('roles_users',
        db.Column('user_id', db.Integer(), db.ForeignKey('user.id')),
        db.Column('role_id', db.Integer(), db.ForeignKey('role.id')))

#class SomeDataObject(db.Model):
#    __tablename__ = 'game_watch'
#    id = db.Column(db.Integer, primary_key=True)
#    url = db.Column(db.String(4096))
#    name = db.Column(db.String(256))
#    active = db.Column(db.Boolean, default=True)
#    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class Role(db.Model, RoleMixin):
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(80), unique=True)

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True)
    password = db.Column(db.String(255))
    active = db.Column(db.Boolean())
    confirmed_at = db.Column(db.DateTime())
    roles = db.relationship('Role', secondary=roles_users,
                            backref=db.backref('users', lazy='dynamic'))
    #data_objects = db.relationship(SomeDataObject, backref='user', lazy=True)
