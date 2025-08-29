import json
import asyncio
import threading
import aiohttp
import paho.mqtt.client as mqtt
from flask import Flask, render_template, jsonify

# --- Flask App ---
app = Flask(__name__)

# --- Supabase Config ---
SUPABASE_URL = "our_url"
SUPABASE_KEY = "our_key"

# --- MQTT Config (HiveMQ) ---
MQTT_BROKER = "our_broker_url"
MQTT_PORT = 8883
MQTT_USER = "LumiVerse"
MQTT_PASS = "our_pass"
MQTT_TOPIC = "esp32/character/emotion"

# --- MQTT Client ---
mqtt_client = mqtt.Client(callback_api_version=2)
mqtt_client.username_pw_set(MQTT_USER, MQTT_PASS)
mqtt_client.tls_set()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
print("Connected to MQTT broker")

# --- Global variable to store last emotion ---
last_emotion = "neutral"

def update_emotion(new_emotion):
    global last_emotion
    last_emotion = new_emotion
    # Publish to HiveMQ with QoS=1 and retain=True
    message = {"emotion": new_emotion}
    mqtt_client.publish(MQTT_TOPIC, json.dumps(message), qos=1, retain=True)
    print("Updated emotion and sent to HiveMQ:", new_emotion)

# --- Flask Routes ---
@app.route('/')
def home():
    return render_template('index.html', emotion=last_emotion)

@app.route('/last_emotion')
def get_last_emotion():
    return jsonify({"emotion": last_emotion})

# --- Polling function ---
async def poll_last_emotion():
    last_time = None
    async with aiohttp.ClientSession(headers={
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Accept": "application/json"
    }) as session:
        while True:
            async with session.get(f"{SUPABASE_URL}?order=timestamp.desc&limit=1") as resp:
                data = await resp.json()
                if data and isinstance(data, list):
                    record = data[0]
                    record_time = record.get("timestamp")
                    emotion = record.get("emotion", "neutral")
                    if record_time != last_time:
                        last_time = record_time
                        update_emotion(emotion)
            await asyncio.sleep(1)

# --- Run polling in background thread ---
def start_polling_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(poll_last_emotion())

# --- Main ---
if __name__ == "__main__":
    # Start polling in background thread
    threading.Thread(target=start_polling_loop, daemon=True).start()
    # Run Flask app
    app.run(debug=True)

