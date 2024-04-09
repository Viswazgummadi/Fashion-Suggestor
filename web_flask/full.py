import sys
from Enigma import *
from detect_gender_implmentation import *
from Clustering import *
import csv
import os
import cv2
from Rating import *
from classifier import *

# GENDER OF INPUT PHOTO

model_path = 'gender_detection_model_knn_k27.model'
clf = joblib.load(model_path)


def process_image_and_write_to_csv(image_path, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        gender_label = detect_gender_from_photo(image_path)
        writer.writerow([os.path.basename(image_path), gender_label])


def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <image_path>")
        return

    input_image = sys.argv[1]
    output_csv = "gender.csv"
    process_image_and_write_to_csv(input_image, output_csv)
    print("Gender detection and CSV creation completed.")

    # BORDERING FOR INPUT PHOTO
    image = cv2.imread(input_image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 30, 190)
    bordered_portion = extract_bordered_portion(image, edges, gray_image)
    cv2.imwrite("bordered_portion.jpg", bordered_portion)
    print("Image processing completed.")

    # COLOR EXTRACTION FOR INPUT PHOTO
    image_path_clustering = "bordered_portion.jpg"
    detect_shirt_colors(image_path_clustering)
    output_file = 'input_color.csv'

    all_colors = detect_shirt_colors(image_path_clustering)
    for i in range(len(all_colors)):
        for j in range(1, len(all_colors[i])):
            for k in range(len(all_colors[i][j])):
                all_colors[i][j][k] = int(all_colors[i][j][k])

    int_colors = [[int(color) for color in colors] for _, colors in all_colors]
    np_colors = np.array(int_colors)

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image', 'color1', 'color2',
                         'color3', 'color4', 'color5'])
        writer.writerow([input_image, np_colors[0], np_colors[1],
                         np_colors[2], np_colors[3], np_colors[4]])

    print("Output was given to:", output_file)

    # RATINGS

    input_file_path = "male.csv"
    output_file_path = "male_output.csv"

    # Load data from existing CSV file
    data = load_data(input_file_path)

    # Generate random ratings for each photo
    data_with_random_ratings = generate_random_ratings(data)

    # Save data with random ratings to a new CSV file
    save_data(output_file_path, data_with_random_ratings)

    print(
        f"Random ratings have been successfully added to the dataset and saved to {output_file_path}.")

    input_file_path = "female.csv"
    output_file_path = "female_output.csv"

    # Load data from existing CSV file
    data = load_data(input_file_path)

    # Generate random ratings for each photo
    data_with_random_ratings = generate_random_ratings(data)

    # Save data with random ratings to a new CSV file
    save_data(output_file_path, data_with_random_ratings)

    print(
        f"Random ratings have been successfully added to the dataset and saved to {output_file_path}.")

    # CLASSIFIER

    input_file = "input_color.csv"
    new_input = read_input(input_file)

    male_csv = "male_output.csv"
    female_csv = "female_output.csv"
    gender_file = "gender.csv"

    male_data = []
    with open(male_csv, 'r') as file:
        csv_reader_male = csv.reader(file)
        count = 0
        for row in csv_reader_male:
            if count > 0:
                male_data.append(row[0])
            count += 1

    female_data = []
    with open(female_csv, 'r') as file:
        csv_reader_female = csv.reader(file)
        count = 0
        for row in csv_reader_female:
            if count > 0:
                female_data.append(row[0])
            count += 1

    X_male, y_male, ratings_male = file_open(male_csv)
    X_female, y_female, ratings_female = file_open(female_csv)

    with open(gender_file, 'r') as file:
        csv_reader_gender = csv.reader(file)
        for row in csv_reader_gender:
            gender = int(row[1])

    # Determine which dataset to use based on gender
    if gender == 0:
        closest_labels = recommend(
            X_male, y_male, ratings_male, new_input, male_data, female_data)
    else:
        closest_labels = recommend(
            X_female, y_female, ratings_female, new_input, male_data, female_data)

    with open("image_names.csv", 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(map(lambda x: [x], closest_labels))

    print("Predicted labels for the three closest data points:", closest_labels)

    output_ratings = "male_ratings.csv"
    generate_ratings(male_csv, output_ratings)

    output_ratings = "female_ratings.csv"
    generate_ratings(female_csv, output_ratings)


if __name__ == "__main__":
    main()
