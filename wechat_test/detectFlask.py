PAGE = '''<!doctype html>
    <title>Chinese Text Detector</title>
    <h1>Chinese Text Detector</h1>
    <form action="" method=post enctype=multipart/form-data>
        <p>
         <label for="image">image</label>
         <input type=file name=file required>
         <input type=submit value=detect>
    </form>
    '''

from flask import Flask,request,jsonify
import cv2
import numpy as np
from mnist1d import detection,load_model

app = Flask('Detector')

@app.route('/', methods=['GET', 'POST'])
# def index():
#     print("GO GO GO!!!")
#     return redirect(url_for('detector'))

@app.route('/detector', methods=['GET', 'POST'])
def detector():
    print("GO GO!!")
    if request.method !='POST':
        return PAGE
    img = request.files['file']
    data = img.read()

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    net = load_model()
    result = int(detection(net=net,img=img))
    
    # print('inference time: ', time.time()-tic)

    return jsonify(msg='success', data={'result': result})


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))