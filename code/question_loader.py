import csv
import json

dataset_location = 'data/question_solution.csv'

def load_json(dataset_location):
    data_csv = open(dataset_location, 'r')
    
    fieldnames = ("Id", "Question", "Solution", "ChatGPT_Answer", "Grade", \
                  "Similar_Questions", "Few-Shot_Answer", "Few-Shot_Evaluation")
    
    reader = csv.DictReader(data_csv, fieldnames)
    for i, row in enumerate(reader):
        data_json = open(f'data/Question_{i+1}.json', 'w')
        json.dump(row, data_json, indent=4)

if __name__ == "__main__":
    load_json(dataset_location)