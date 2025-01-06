python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
python web/manage.py makemigrations
python web/manage.py migrate
start http://127.0.0.1:8000/models/upload/
python web/manage.py runserver
cmd /k

