# Flask App for an Instruct TinyLLM

## Installation
 - Clone the repository `git clone https://github.com/jamesnogra/tiny-llm-flask-app.git`
 - Go inside the cloned repository
 - Then `pip install -r requirements.txt`
 - Create a `.env` file with the contents `TOKENS` and `PORT`
   - Sample `TOKENS` is `TOKENS=["jamestoken1","anotherlongtoken"]`

## Gunicorn Information
 - After pulling from `master`, make sure to `systemctl daemon-reload` and `systemctl restart tiny_llm_flask.service`
 - To check the status of the service, run `systemctl status tiny_llm_flask.service`
 - The Gunicorn service is at `/etc/systemd/system/tiny_llm_flask.service` while the Gunicorn config is at `gunicorn.config.py`
 - The logs are located in the `logs` directory same with the project folder

 ## Running Locally
 - Run `python app.py`