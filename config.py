import os

SQLALCHEMY_DATABASE_URI = 'sqlite:////data/image_search.db'
FLASK_ADMIN_SWATCH = 'flatly'
SCHEDULER_API_ENABLED = False
SECRET_KEY = os.environ["FLASK_SECRET_KEY"]
SECURITY_REGISTERABLE = False

SECURITY_PASSWORD_SALT = os.environ['SECURITY_PASSWORD_SALT']
SECURITY_EMAIL_SENDER = "Image Search"
SECURITY_CONFIRMABLE = True
SECURITY_REGISTERABLE = True
SECURITY_RECOVERABLE = True
SECURITY_CHANGEABLE = True
UPLOADED_PHOTOS_DEST = os.environ["UPLOAD_DIR"]

MAIL_SERVER = 'smtp.gmail.com'
MAIL_PORT = 465
MAIL_USE_TLS = False
MAIL_USE_SSL = True
MAIL_USERNAME = "mscherb.server@gmail.com"
MAIL_PASSWORD = os.environ['MAIL_PASSWORD']
MAIL_DEFAULT_SENDER = "Image Search"
MAIL_MAX_EMAILS = 10
MAIL_ASCII_ATTACHMENTS = False
