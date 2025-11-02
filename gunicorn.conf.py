from dotenv import load_dotenv
import os

# Load variables from .env
load_dotenv()

PORT = int(os.getenv("PORT", 5000))

# Bind to all interfaces, port 5000
bind = f"0.0.0.0:{PORT}"

# Number of worker processes (adjust to number of CPU cores)
workers = 2

# Automatically restart workers if they consume too much memory
max_requests = 1000
max_requests_jitter = 50

# Timeout (seconds before a worker is killed and restarted)
timeout = 30

# Log files
accesslog = "/var/www/html/test_small_llm/logs/access.log"
errorlog = "/var/www/html/test_small_llm/logs/error.log"

# Enable logging to stdout (optional)
capture_output = True

# Display the app’s environment (optional)
raw_env = ["FLASK_ENV=production"]

# Graceful reload on SIGHUP
reload = False