from flask import Flask, request, jsonify, render_template
from rag_core import initialize_rag_chain

# Initialize the Flask app
app = Flask(__name__)

# Load the RAG chain once when the application starts
print("Initializing RAG chain...")
rag_query_func = initialize_rag_chain()
print("RAG chain initialized successfully.")

# Route for the main chat page
@app.route("/")
def index():
    return render_template("index.html")

# Route to handle chat messages
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    if not user_input:
        return jsonify({"reply": "Please enter a message."})
    
    # Get the response from your RAG function
    response = rag_query_func(user_input)
    
    return jsonify({"reply": response})

# Run the app
if __name__ == "__main__":
    # Use host='0.0.0.0' to make it accessible on your local network
    app.run(host="0.0.0.0", port=5001, debug=False)