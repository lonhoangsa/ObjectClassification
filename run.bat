call venv\Scripts\activate
python web/manage.py migrate
start http://127.0.0.1:8000/models/upload/
python web/manage.py runserver
