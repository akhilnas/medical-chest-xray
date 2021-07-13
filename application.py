from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from sampler import image_classifier

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

application = Flask(__name__)
application.secret_key = "secret key"
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    '''
    Function to check if filename has a valid extension
    '''
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@application.route('/')
def home():
    return render_template('index.html')

# Route to upload image
@application.route('/', methods = ['POST'])
def upload_image():
    if 'file' not in request.files:
       flash('No fiel part')
       return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for upload')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(application.config['UPLOAD_FOLDER'], filename))
        flash('Image successfully uploaded')
        class_name = image_classifier()
        return render_template('index.html', filename=filename, category = class_name)
    else:
        flash('Allowed image types are png , jpg and jpeg.')
        return redirect(request.url)

@application.route('/display/<filename>')
def display_image(filename):
    print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
    
if __name__ == "__main__":
    application.run(debug=True)

