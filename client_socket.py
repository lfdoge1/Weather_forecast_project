import socket

# ----------------------------
# Server address configuration
# ----------------------------
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8888

# ----------------------------
# reate socket and connect to server
# ----------------------------
client = socket.socket()

try:
    client.connect((SERVER_HOST, SERVER_PORT))
    print(" Connected to chatbot server.")
except Exception as e:
    print(f"❌ Unable to connect to server: {e}")
    exit()

# ----------------------------
# Input prompt for user
# ----------------------------
print("=" * 70)
print("️  To simulate, today is 2023-09-01 in Delhi and 1986-08-29 in Coventry.")
print("=" * 70)
print("Hello, I'm Oracle Chatbot. I can help you check the weather forecast for the next 7 days in Delhi and Coventry.")
print("You can try asking questions like:")
print("  - What's the weather like in Delhi tomorrow?")
print("  - How's the weather in Coventry 3 days from now?")
print("Type 'exit' anytime to end the chat.\n")

# ----------------------------
# Main chat loop
# ----------------------------
while True:
    try:
        message = input(" You: ").strip()

        if message.lower() == "exit":
            print("👋 Disconnected.")
            break

        if not message:
            continue

        # send message
        client.send(message.encode())

        # reoly
        reply = client.recv(1024).decode(errors="ignore")
        print("Oracle Chatbot:", reply)

    except Exception as e:
        print(f"⚠️ Error during communication: {e}")
        break

# ----------------------------
# Close connection
# ----------------------------
client.close()
