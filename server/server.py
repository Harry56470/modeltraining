from flask import Flask,render_template,request
from ultralytics import YOLO
app=Flask(__name__)
@app.route('/') 
def home(): 
    return "<h1>Welcome to the Page!</h1><p>This is a simple Flask app.</p>"

@app.route('/upload',methods=["GET","POST"])
def upload():
    if request.method == "POST":#upload video
        file=request.files["file"]
        file.save("image.jpeg")
        analyze("image.jpeg")
    return render_template("index.html")

def analyze(image):
    model=YOLO("models/yolov8n.pt")
    results=model.predict(image,save=True)


if __name__ == "__main__":
    app.run(debug=True)