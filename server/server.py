from flask import Flask,render_template,request
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials,firestore,storage
import os
import json

service_account_path = os.path.join(
    os.path.dirname(__file__),
    "soccer-app-2d09b-firebase-adminsdk-fbsvc-0ffbf37e55.json",
)

with open(service_account_path) as f:
    project_id = json.load(f)["project_id"]

cred = credentials.Certificate(service_account_path)
firebase_admin.initialize_app(cred, {
    "storageBucket": f"{project_id}.firebasestorage.app",
})
db = firestore.client()
bucket = storage.bucket()

app=Flask(__name__)
@app.route('/') 
def home(): 
    return "<h1>Welcome to the Page!</h1><p>This is a simple Flask app.</p>"

@app.route('/upload',methods=["GET","POST"])
def upload():
    if request.method == "POST":#upload video
        file=request.files["file"]
        uid=request.form.get("user_id")
        name=request.form.get("name")
        file.save("image.jpeg")
        analyze("image.jpeg",uid,name)
    return render_template("index.html")

def analyze(image,uid,name):
    model=YOLO("models/yolov8n.pt")
    results=model.predict(image,save=True)
    upload_image=bucket.blob(f"{uid}/{name}.jpg")
    upload_image.upload_from_filename("runs/detect/predict/image.jpg")
    upload_image.make_public()
    doc_ref=db.collection("videos").document()
    doc_ref.set(
        {
            "name": name,
            "url":upload_image.public_url,
            "uid":uid,
            "dateCreated":firestore.SERVER_TIMESTAMP
        }
    )


if __name__ == "__main__":
    app.run(debug=True)