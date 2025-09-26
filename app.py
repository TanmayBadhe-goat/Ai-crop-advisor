from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({
        'message': 'KrishiMitra API Running', 
        'status': 'OK',
        'version': 'minimal'
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    user_msg = data.get('message', '')
    
    # Simple fallback response for now
    response = f"🌾 Thank you for your question about: '{user_msg}'. The KrishiMitra chatbot is currently in minimal mode. Please try again later for full AI assistance!"
    
    return jsonify({
        'success': True,
        'response': response,
        'source': 'minimal_fallback'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
