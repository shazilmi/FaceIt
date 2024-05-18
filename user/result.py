from flask import Blueprint, render_template, request, session

results = Blueprint('result', __name__)

@results.route('/result')
def result():
    asd = session['output']
    stddev = session['stddev']
    if asd == 0:
        thestring = 'You are unlikely to have ASD, with a value of:' + str(stddev) + '.'
    else:
        thestring = 'You are likely to have ASD, with a value of:' + str(stddev) + '.'
    return render_template('result.html', thestring = thestring)
