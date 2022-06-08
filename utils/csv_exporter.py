import csv
import os
from typing import List


def export(base_dir, filename, data):
    filepath = os.path.join(base_dir, filename)
    print(filepath)
    with open(file=filepath, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        for row in data:
            writer.writerow(row)

def export_csv(data: 'list[dict]', filepath: str, fields: 'list[str]'):
    with open(file=filepath, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()

        for d in data:
            # print(d)
            # print(d.values())
            writer.writerow(d)

def import_csv(filepath: str) -> List[dict]:
    with open(file=filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        
        rows = []
        for row in reader:
            # print(row)
            if 'acc' in row:
                row['acc'] = eval(row['acc'])
            # print(acc[1])
            rows.append(row)
    return rows

def import_simple_csv(filepath: str) -> List[dict]:
    
    with open(file=filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        
        rows = []
        for row in reader:
            rows.append(row)
            break
    
    return rows