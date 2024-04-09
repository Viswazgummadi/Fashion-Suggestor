import csv
import random

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            data.append(row)
    return data

def save_data(file_path, data):
    with open(file_path, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(data)

def generate_random_ratings(data):
    for row in data:
        rating = random.randint(1, 5)  # Generate random rating between 1 and 5
        row.append(rating)
    return data


