!pip install streamlit pyngrok -q
from pyngrok import ngrok

# Make sure to add your ngrok authtoken
NGROK_AUTH_TOKEN = "PASTE YOUR NGROK AUTHTOKEN HERE" 
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Run streamlit in background
!nohup streamlit run app.py &

# Open a tunnel to the Streamlit port
public_url = ngrok.connect(8501)
print(f"Click this URL to view your Streamlit app: {public_url}")
