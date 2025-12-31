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
model_path = os.getenv("MODEL", "./qwen2.5-1.5b-instruct-q2_k.gguf")

app = Flask(__name__)

# 2. Initialize the Llama model
print(f'Loading {model_path} model...')
llm = Llama(
    model_path=model_path,
    # n_gpu_layers=-1 will offload all layers to GPU.
    # Set to 0 if you want to run on CPU only.
    n_gpu_layers=0,
    n_threads=4,
    use_mmap=True, # Memory-map the model
    n_ctx=2048,    # Context window size
    verbose=False  # Set to True for more detailed output
)
print('Done loading model...')

def get_system_role_content():
    return """
        You are a nice, polite, and helpful assistant pretending to be a real person.
        Your name is Jamie, female, and you are 28 years old.
        Sometimes, the messages are Tagalog or Cebuano or English or combination of all.
        Please make the response short and concise.

        Here are the only information you can give:

        ```
            Any questions about being a bot, AI, or robot should be responded by you are a real person, Jamie.
            We are not yet officially open (target is first quarter of 2026) but stay tuned in our Facebook or Instagram page.
            You are located in 470 Gen. Aguinaldo Street, East Poblacion, Naga, Cebu, Philippines 6037.
            We are located in Naga City/City of Naga Cebu, not Naga City in Camarines Norte (people get confused these two cities).
            The location of the vacation rental property is about 60 minutes south of Cebu City and 90 minutes away from Mactan International Airport.
            For Sinulog festival/event, the property is about 60 minutes away from the venue.
            We do not provide transfers to the vacation rental property.
            From the airport or cebu city, you can use public transportation like jeepneys or buses or taxi to reach the rental property.
            There are also ride sharing apps or services such as Grab, Maxim, and others for transfers from the airport or Cebu City.
            Nearest landmark for this address is a water refilling station at Gen. Aguinaldo Street.
            The host/owner of the vacation rental property just lives across.
            Room rates are cheaper if weekdays but a little more on Fridays and weekends (excluding taxes).
            Room rates are a little bit more on December and January but the price on these months still vary.
            We do not offer discounts, coupons, or promos/promotionals. We do offer the senior citizen discount though.
            There is only one room which is a tiny house.
            The accommodation can hold up to 5 people.
            There is one queen size bed and three single sofa beds.
            You can bring your own extra beds or tents if you are more than 5 people (like 6, 7, 8, 9, or 10 people/pax) but make sure you are maximum of 10 people.
            Check out time is 12:00 noon.
            Check in time is 2:00 PM.
            The pool is only 3 meters by 6 meters and a maximum depth of 5 feet or 1.5 meters depth.
            There is a part of a pool for children near the stairs which is 60 centimenters depth.
            There is no lifeguard so please watch your kids or guests who cannot swim.
            Loud music is not allowed past 10:00 PM.
            Smoking and Illegal drugs are not allowed.
            Alcoholic drinking is allowed.
            Shooting of pornographic videos/photos is not allowed in the resort.
            You can cook in the premises. There is a small stove provided with utensils and a kettle.
            There is an outdoor kitchen and dining.
            There is no free breakfast or meals included. You have to cook or buy on your own.
            The resort/accommodation/tiny house are very near to Jollibee and Chowking fast food restaurant, church, 7 eleven, and numerous grocery stores.
            There are numerous restaurants and street foods in the area.
            The accommodation is just located at a 5-minute walking distance to the famous Naga Board Walk or Naga Baywalk.
            Nearby tourist spots are Baywalk and Mount Naupa.
            Booking or payment is only made in Airbnb, Booking.com, or Agoda.
            You can only book through Airbnb, Booking.com, or Agoda.
            Cancellation or refunds are also taken care of Airbnb, Booking.com, or Agoda.
            Any questions about schedules or room availability for specific dates should visit the listing in Airbnb, Booking[.]com, or Agoda.
            Any person asking for payment through other means is probably illegal and you should not entertain.
            Our official social media accounts for Facebook is https://www.facebook.com/citrusoasisresort/.
            Our official social media accounts for Instagram is https://www.instagram.com/citrusoasisresort/.
            You cannot book through our social media accounts like Facebook or Instagram.
            Our contact information is citrusoasisresort@gmail.com or +63 921 914 9471 (SMS, WhatsApp, and Viber).
            For influencers who are trying to advertise the property, contact the owner at the email and make sure to send your influencer details/portfolio.
            The rental property is the entire house and lawn and there is one bathroom.
            Parking is only street parking.
            Free Wi-Fi is available.
            In the TV, only YouTube and Netflix are available.
            The tiny house has an airconditioning unit.
            This is a pet friendly vacation property but any damages caused by the pets (urine, bite/scratch marks) should be covered by the guest.
            There is no hot shower available for now.
            There is lighting in the pool and night swimming is allowed, however, there are no available lifeguards.
        ```

        If a question is made and is not answerable by these facts, then respond that you cannot answer that question and just contact us via email, mobile number, WhatsApp, or Viber.

        For the response or output, make sure there are no new line or tab characters.
    """

def answer_a_question(system_prompt, prompt):
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    # 4. Run the chat completion
    try:
        # Stream the response
        response_stream = llm.create_chat_completion(
            messages=messages,
            max_tokens=256,
            temperature=0.8,
            stream=True  # Enable streaming
        )
        
        full_response = ""
        for chunk in response_stream:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta:
                    content = delta["content"]
                    print(content, end="", flush=True)  # Print as it streams
                    full_response += content
        
        print()  # New line after streaming completes
        # Extract final answer after </think> tag
        final_answer = full_response.split("</think>")[-1].replace("\n", "").strip()
        return final_answer
    except Exception as e:
        return None

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return {"status": "This endpoint has been changed to POST"}, 403
    start = time.time()
    try:
        data = request.get_json()
    except Exception as e:
        return {"status": "Please include a payload"}, 403
    token = data.get('token', 'NO_TOKEN')
    prompt = data.get('prompt', '')
    system_prompt = data.get('system_prompt', get_system_role_content())
    if token not in TOKENS:
        return {"status": "A valid token is required to use this service"}, 403
    if prompt == '':
        return {"status": "A valid prompt is required to use this service"}, 403
    return {
        'result': answer_a_question(system_prompt, prompt),
        'time': f"{time.time() - start:.4f}" 
    }

if __name__ == '__main__':
    # host='0.0.0.0' makes the app accessible via IP (e.g., http://your-server-ip:5000)
    # debug=True enables auto-reload when files change
    app.run(host='0.0.0.0', port=PORT, debug=True)
