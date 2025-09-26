from app import app

# This is the entry point for Vercel
# Vercel will automatically detect this as a Flask app
if __name__ == "__main__":
    app.run(debug=True)
