import os
import csv
import sys
import random
from flask import Flask, request, render_template, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import subprocess

app = Flask(__name__)

# Set the secret key
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# Define the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if a filename has an allowed extension


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

# Add the check_favorite route to check if an image is in favorites


@app.route('/check-favorite')
def check_favorite():
    image_src = request.args.get('imageSrc')
    with open('fav.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if image_src in row:
                return {'isFavorite': True}
    return {'isFavorite': False}


@app.route('/check-ratings')
def check_ratings():
    image_src = request.args.get('imageSrc')
    with open('ratings.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if image_src in row:
                # Return the rating value if image is found
                return {'rating': int(row[1])}
    # Return 0 if image is not found in ratings list
    return {'rating': 0}
# Add the toggle_favorite route to handle adding/removing images from favorites


@app.route('/check-rating-changes')
def check_rating_changes():
    ratings = []
    with open('ratings.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            ratings.append({'imageSrc': row[0], 'rating': int(row[1])})
    return jsonify(ratings)


@app.route('/toggle-favorite', methods=['POST'])
def toggle_favorite():
    image_src = request.form['imageSrc']
    # Read existing favorites
    with open('fav.csv', 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    found = False
    for row in rows:
        if image_src in row:
            rows.remove(row)
            found = True
            break

    if not found:
        rows.append([image_src])

    # Write updated favorites
    with open('fav.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

    return jsonify({'success': True})  # Return success response
# Add the toggle_rating route to handle adding/removing images from ratings


@app.route('/toggle-rating', methods=['POST'])
def toggle_rating():
    image_src = request.form['imageSrc']
    rating = int(request.form['rating'])
    # Read existing ratings
    with open('ratings.csv', 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    found = False
    for row in rows:
        if image_src in row:
            # Update the rating value for the image
            row[1] = str(rating)
            found = True
            break

    if not found:
        # If the image is not found, append a new row with the image and rating
        rows.append([image_src, str(rating)])

    # Write updated ratings
    with open('ratings.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

    return jsonify({'success': True})


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dev.html')
def dev():

    return render_template('dev.html')


@app.route('/favs.html')
def favs():
    image_paths = []
    with open('fav.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            image_paths.extend(row)

    # Get the path to the images directory
    images_dir_male = os.path.join(app.root_path, 'static', 'images', 'male')
    images_dir_female = os.path.join(
        app.root_path, 'static', 'images', 'female')
    images_dir_shirts = os.path.join(
        app.root_path, 'static', 'images', 'shirts')

    return render_template('favs.html', image_paths=image_paths, images_dir_male=images_dir_male, images_dir_female=images_dir_female, images_dir_shirts=images_dir_shirts)


@app.route('/work.html')
def work():
    return render_template('work.html')


@app.route('/crew.html')
def crew():
    return render_template('crew.html')


@app.route('/sugg.html')
def sugg():
    image_names = []
    with open('image_names.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            image_names.extend(row)

    # Get the path to the images directory
    images_dir = os.path.join(app.root_path, 'images')

    return render_template('sugg.html', image_names=image_names, images_dir=images_dir)


@app.route('/cart.html')
def cart():
    # Get list of all image filenames in static/images folder
    images_dir = os.path.join(app.static_folder, 'images', 'shirts')
    image_files = os.listdir(images_dir)

    # Ensure there are enough images to sample from
    if len(image_files) >= 10:
        # Randomly select 10 images
        selected_images = random.sample(image_files, 10)
    else:
        # If there are not enough images, select all available images
        selected_images = image_files

    return render_template('cart.html', image_names=selected_images, images_dir=images_dir)

# Route to handle image upload and processing


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        # If no file is uploaded, print a message to the console
        print("No file uploaded")
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        # If the filename is empty, return without doing anything
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Call full.py with the uploaded image
        subprocess.run([sys.executable, 'full.py', os.path.join(
            app.config['UPLOAD_FOLDER'], filename)])

        # Update image_names.csv after running full.py
        with open('image_names.csv', 'r') as file:
            image_names = file.readlines()
            # Process image_names if needed
        return redirect('/sugg.html')  # Redirect to sugg.html after upload


if __name__ == '__main__':
    app.run(debug=True)
