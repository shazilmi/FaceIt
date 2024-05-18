from flask import Blueprint, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from functions.file import check_extension, preprocess, save_processed
from functions.model import get_prediction
import os

take_tests = Blueprint('take_test', __name__)

@take_tests.route('/taketest', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            filename = secure_filename(file.filename)
            if check_extension(filename):
                filepath = os.path.join('static/uploads', filename)
                file.save(filepath)
                processed = preprocess(filepath)
                save_processed(processed, filepath)
                output, stddev = get_prediction(processed)
                session['output'] = output
                session['stddev'] = stddev
                return redirect('result')
            else:
                return 'Need an image to be uploaded.'
    return render_template('upload.html')