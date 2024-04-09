import csv
import numpy as np
import pandas as pd

def file_open(file):
    outs = []
    ratings = []
    c1 = []
    c2 = []
    c3 = []

    with open(file, 'r') as file:
        csv_reader = csv.reader(file)
        
        for row in csv_reader:
            
            outs.append(row[0])
            ratings.append(int(row[6]))
            cc1 = row[1].strip('[]').split()
            c1.append([int(element) for element in cc1])
            cc2 = row[2].strip('[]').split()
            c2.append([int(element) for element in cc2])
            cc3 = row[3].strip('[]').split()
            c3.append([int(element) for element in cc3])
        

    outs = np.array(outs)
    ratings = np.array(ratings)
    c1 = np.array(c1)
    c2 = np.array(c2)
    c3 = np.array(c3)

    X = np.hstack((c1, c2, c3))
    y = np.array(outs)
    
    return X,y,ratings

def recommend(X, y, ratings, new_input, male_data, female_data):
    distances = [cal_dis(new_input, x) for x in X]
    weighted_distances = distances * (1 / ratings)
    closest_indices_dist = np.argsort(weighted_distances)[:5]
    
    # Step 2: Select 5 samples based only on ratings
    closest_indices_ratings = np.argsort(ratings)[:5]
    
    # Step 3: Randomly select 5 samples from both male and female datasets
    male_indices = np.random.choice(np.arange(len(male_data)), 3, replace=False)
    female_indices = np.random.choice(np.arange(len(female_data)), 2, replace=False)
    
    # Combine all selected indices
    all_indices = np.concatenate((closest_indices_dist, closest_indices_ratings, male_indices, female_indices))
    
    # Filter indices to ensure they are within bounds
    valid_indices = [i for i in all_indices if i < len(y)]
    
    # Get the corresponding labels
    closest_labels = y[valid_indices]
    
    return closest_labels
        

def cal_dis(inp1, inp2):
    inp1 = np.array(inp1, dtype=int)
    d_x = np.sqrt((inp1[0][0] - inp2[0])**2 + (inp1[0][1] - inp2[1])**2 + (inp1[0][2] - inp2[2])**2)
    d_y = np.sqrt((inp1[0][3] - inp2[3])**2 + (inp1[0][4] - inp2[4])**2 + (inp1[0][5] - inp2[5])**2)
    d_z = np.sqrt((inp1[0][6] - inp2[6])**2 + (inp1[0][7] - inp2[7])**2 + (inp1[0][8] - inp2[8])**2)
    dis = np.sqrt(d_x**2 + d_y**2 + d_z**2)
    return dis


def read_input(file):
    new1 = []
    new2 = []
    new3 = []
    new4 = []
    new5 = []
    new_input = []
    with open(file, 'r') as file:
        csv_reader_input = csv.reader(file)
        count = 0
        for row in csv_reader_input:
            if count > 0:
                newi_1 = row[1].strip('[]').split()
                new1.append([int(element) for element in newi_1])
                newi_2 = row[2].strip('[]').split()
                new2.append([int(element) for element in newi_2])
                newi_3 = row[3].strip('[]').split()
                new3.append([int(element) for element in newi_3])
                newi_4 = row[4].strip('[]').split()
                new4.append([int(element) for element in newi_4])
                newi_5 = row[5].strip('[]').split()
                new5.append([int(element) for element in newi_5])
            count += 1


    new_input= np.hstack((new1,new2,new3,new4,new5))
    
    return new_input



def combine_ratings(male_file, female_file, output_file):
    # Read male and female output files
    male_df = pd.read_csv(male_file)
    female_df = pd.read_csv(female_file)
    
    # Extract image names and ratings
    male_image_names = male_df.iloc[:, 0].tolist()
    female_image_names = female_df.iloc[:, 0].tolist()
    
    male_ratings = male_df.iloc[:, 6].tolist()
    female_ratings = female_df.iloc[:, 6].tolist()
    
    # Combine image names and ratings
    all_image_names = male_image_names + female_image_names
    all_ratings = male_ratings + female_ratings
    
    # Write to ratings.csv
    ratings_df = pd.DataFrame({'Image Name': all_image_names, 'Ratings': all_ratings})
    ratings_df.to_csv(output_file, index=False)

            