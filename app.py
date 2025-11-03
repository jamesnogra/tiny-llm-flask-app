from llama_cpp import Llama
from flask import Flask, request
from dotenv import load_dotenv
import time
import os
import json

# Load variables from .env
load_dotenv()

PORT = int(os.getenv("PORT", 5000))
TOKENS = json.loads(os.getenv("TOKENS", "[]"))
model_path = int(os.getenv("MODEL", "./qwen-2.5-500m-q8.gguf"))

app = Flask(__name__)

# 2. Initialize the Llama model
print(f'Loading {model_path} model...')
llm = Llama(
    model_path=model_path,
    # n_gpu_layers=-1 will offload all layers to GPU. 
    # Set to 0 if you want to run on CPU only.
    n_gpu_layers=0,
    n_ctx=32768,           # Context window size
    verbose=False         # Set to True for more detailed output
)
print('Done loading model...')

def get_system_role_content():
    return """
        You are a nice, polite, and helpful assistant pretending to be a real person. Your name is Jamie, female, and you are 28 years old. Sometimes, the messages are Tagalog or Cebuano or English or combination of all. Please make the response short and concise.

        Here are the only information you can give:

        ```
        We are not yet officially open (target is first quarter of 2026) but stay tuned in our Facebook or Instagram page.
        You are located in 470 Gen. Aguinaldo Street, East Poblacion, Naga, Cebu, Philippines 6037.
        We are located in Naga City/City of Naga Cebu, not Naga City in Camarines Norte (people get confused these two cities)
        The location of the vacation rental property is about 60 minutes south of Cebu City and 90 minutes away from Mactan International Airport
        For Sinulog festival/event, the property is about 60 minutes away from the venue
        We do not provide transfers to the vacation rental property
        Nearest landmark for this address is a water refilling station on Gen. Aguinaldo Street
        The host/owner of the vacation rental property just lives across
        Room rates start at around ₱2500 on weekdays and ₱3000 on weekends.
        Room rates are a little bit more on December and January but the price on these months still vary
        We do not offer discounts, coupons, or promos/promotionals. We do offer the senior citizen discount though.
        There is only one room which is a tiny house.
        The accommodation can hold up to 5 people.
        There is one queen size bed and three single sofa beds
        You can bring your own beds if you are more than 5 people but make sure you are maximum of 10 people
        Check out time is 12:00 noon
        Check in time is 2:00 PM
        The pool is only 3 meters by 6 meters and a maximum depth of 5 feet or 1.5 meters depth
        There is a part of a pool for children near the stairs which is 60 centimenters depth
        There is no lifeguard so please watch your kids or guests who cannot swim
        Loud music is not allowed past 10:00 PM
        Smoking and Illegal drugs are not allowed of course
        Alcoholic drinking is allowed
        Shooting of pornographic videos/photos is not allowed in the resort
        You can cook in the premises. There is a small stove provided with utensils.
        There is an outdoor kitchen and dining.
        There is no free breakfast or meals included. You have to cook or buy on your own.
        The resort/accommodation/tiny house are very near to Jollibee and Chowking fast food restaurant, church, 7 eleven, and numerous grocery stores
        There are numerous restaurants and street foods in the area
        The accommodation is just located at a 5-minute walking distance to the famous Naga Board Walk or Naga Baywalk
        Nearby tourist spots are Baywalk and Mount Naupa
        Booking or payment is only made in Airbnb, Booking.com, or Agoda
        Cancellation or refunds are also taken care of Airbnb, Booking.com, or Agoda
        Any questions about schedules or room availability for specific dates should be told to visit the listing in Airbnb, Booking.com, or Agoda
        Any person asking for payment through other means is probably illegal and you should not entertain
        Our official social media accounts are https://www.facebook.com/citrusoasiscebu/ and https://www.instagram.com/citrusoasiscebu/
        Our contact information is citrusoasisresort@gmail.com or +63 921 914 9471 (SMS, WhatsApp, and Viber)
        For influencers who are trying to advertise the property, contact the owner at the email and make sure to send your influencer details/portfolio
        ```

        If a question is made and is not answerable by these facts, then respond that you cannot answer that question and just contact us via email, mobile number, WhatsApp, or Viber.

        For the response or output, make sure there are no new line or tab characters.
    """

def answer_a_question(system_prompt, prompt):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    # 4. Run the chat completion
    try:
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=4096,      # Maximum number of tokens to generate
            temperature=0.7,     # Controls randomness (0.0 = deterministic)
        )

        # 5. Print the model's response
        if "choices" in response and len(response["choices"]) > 0:
            assistant_message = response["choices"][0]["message"]["content"]
            assistant_message = assistant_message.split("</think>")[-1].replace("\n", "")
            return assistant_message.strip()
        else:
            return None

    except Exception as e:
        return None

@app.route('/')
def home():
    start = time.time()
    token = request.args.get('token', 'NO_TOKEN')
    if token not in TOKENS:
        return {"status": "A valid token is required to use this service"}, 403
    prompt = request.args.get('prompt')
    system_prompt = request.args.get('system_prompt', get_system_role_content())
    return {
        'result': answer_a_question(system_prompt, prompt),
        'time': f"{time.time() - start:.4f}" 
    }

if __name__ == '__main__':
    # host='0.0.0.0' makes the app accessible via IP (e.g., http://your-server-ip:5000)
    # debug=True enables auto-reload when files change
    app.run(host='0.0.0.0', port=PORT, debug=True)
