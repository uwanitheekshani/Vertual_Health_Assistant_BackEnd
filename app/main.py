from flask import Flask, request, jsonify
from flask_cors import CORS
from train_model import get_bot_response  # Import your actual bot response logic
from flask_mysqldb import MySQL

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, 
     allow_headers=["*"],
     allow_methods=["*"],
     expose_headers=["*"])
# Configure MySQL connection
app.config['MYSQL_HOST'] = '127.0.0.1'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '1234'
app.config['MYSQL_DB'] = 'health_assistant'

mysql = MySQL(app)

@app.route('/')
def index():
    return "Welcome to the Virtual Health Assistant API"

@app.route('/api/get_response', methods=['POST'])
def get_response():
    user_input = request.json.get('user_input')
    bot_response = get_bot_response(user_input, 'health_assistant_model.joblib', 'G:\Volume E\Virtual_Health_Assistant(BE)\dataset\med_2.csv')
    return jsonify({'bot_response': bot_response})


@app.route('/signup', methods=['POST'])
def signup():
    # Get signup data from request
    signup_data = request.json
    username = signup_data.get('username')
    password = signup_data.get('password')
    email = signup_data.get('email')
    firstName = signup_data.get('firstName')

    # Validate signup data (add your validation logic here)
    if not username or not password or not email or not firstName:
        return jsonify({'error': 'All fields are required'}), 400

    # Check if user already exists
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users WHERE username = %s", (username,))
    existing_user = cur.fetchone()
    if existing_user:
        cur.close()
        return jsonify({'response': 1, 'error': 'Username already exists'}), 200

    # Insert new user into the database
    cur.execute("INSERT INTO users (username, password, email, firstName) VALUES (%s, %s, %s, %s)",
                (username, password, email, firstName))
    mysql.connection.commit()
    cur.close()

    return jsonify({'response': 0, 'message': 'User signed up successfully'}), 201


@app.route('/login', methods=['POST'])
def login():
    # Get login data from request
    login_data = request.json
    username = login_data.get('username')
    password = login_data.get('password')

    # Validate login data (add your validation logic here)
    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400

    # Check if user exists and password is correct
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
    user = cur.fetchone()
    cur.close()

    if not user:
        return jsonify({'error': 'Invalid username or password'}), 401  # HTTP 401 Unauthorized status code

    # Handle successful login
    return jsonify({'message': 'Login successful'}), 200


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=80)