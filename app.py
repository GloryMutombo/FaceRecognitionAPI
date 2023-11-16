from flask import Flask, request, jsonify
from processor import predict_sim
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

TRANSIT_FOLDER = 'transit'
app.config['TRANSIT_FOLDER'] = TRANSIT_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(TRANSIT_FOLDER):
    os.makedirs(TRANSIT_FOLDER)






@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/confirm', methods=['POST'])
def confirm_image():

    try:
        # Check if the POST request has a file part
        if 'image' not in request.files or 'student_number' not in request.form:
            return jsonify({'error': 'Image and student_number are required'}), 400

        image_file = request.files['image']
        student_number = request.form['student_number']

        # Check if the file is empty
        if image_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Create a subdirectory for each student number
        student_upload_folder = os.path.join(app.config['TRANSIT_FOLDER'], student_number)
        if not os.path.exists(student_upload_folder):
            os.makedirs(student_upload_folder)

        # Save the image to the student's subdirectory
        image_path = os.path.join(student_upload_folder, image_file.filename)
        image_file.save(image_path)

        sim = predict_sim(image_path, student_number)

        deleteTransitFolder(image_path)

        # Use jsonify to convert the dictionary to a JSON response
        response = jsonify(sim)

        # Optionally, you can set the response status code (200 by default)
        response.status_code = 200

        return response

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

def deleteTransitFolder(file_path):
    if not file_path:
        return jsonify({'error': 'File path not provided'}), 400

    try:
        # Use os.remove to delete the file
        os.remove(file_path)
        return 'Found and deleted'
    except FileNotFoundError:
        return jsonify({'error': f'File {file_path} not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
