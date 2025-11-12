Install requirements:
pip install -r requirements.txt

Fetch the data:
python fetch_data.py

Clean up bad images and add a couple random ones (and specially things that are not a Xis) on 'other' folder

Train the model:
python main.py train

Test the model:
python main.py test test.jpg
