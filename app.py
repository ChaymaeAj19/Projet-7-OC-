from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    # Définir une variable à passer à la page HTML
    message = "Bienvenue sur ma page dynamique avec Flask !"
    
    # Passer la variable 'message' au template HTML
    return render_template('index.html', message=message)

if __name__ == '__main__':
    app.run(debug=True)
