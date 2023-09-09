#this is our first flask app file. 
#The main point from where the flask app operates.

#importing flask as our library
#importing render_template to render our pages
#importing os for file handling and manipulation
#importing OCR library from deeplearing module
from flask import Flask, render_template, request
import os
from deeplearning import OCR


#webserver Gateway interface

app= Flask(__name__)

BASE_PATH= os.getcwd()
UPLOAD_PATH=os.path.join(BASE_PATH,'static/upload/')


#Creation of our main app below, providing GET and POST methods for file upload and result download
@app.route('/',methods=['POST','GET'])
def index():
        
        #if request is POST, then upload the file, save it locally also, provide this image to the object detection function, after that do OCR on the roi image, and at last, render this result to the result page
        if request.method=='POST':
              upload_file=request.files['image-name']
              filename=upload_file.filename
              path_save=os.path.join(UPLOAD_PATH,filename)
              upload_file.save(path_save)
              text = OCR(path_save,filename)
              return render_template('index.html',upload=True,upload_image=filename,text=text)    

        #if this is not done, simply render the homepage.
        return render_template('index.html',upload=False)    


if __name__=="__main__":
    app.run(debug=True)


