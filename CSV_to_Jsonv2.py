#imports
import csv
import json

#open CSV
FILE_NAME = "FILE_NAME"

with open(f'{FILE_NAME}.csv', 'r', encoding='utf-8') as csvfile:

    #read CSV into dictionary & create data list
    r = csv.DictReader(csvfile)
    data = []
    
    #For every row in CSV add to dictionary
    for row in r:
        data.append(dict(row))

#Throw it in JSON
print('dump')
with open(f'{FILE_NAME}.json', 'w') as jsonfile:
    json.dump(data, jsonfile)