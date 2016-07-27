from tempfile import NamedTemporaryFile
import shutil
import csv

filename = 'data/character-predictions_edited_onehot.csv'
tempfile = NamedTemporaryFile(delete=False)

with open(filename, 'rb') as csvFile, tempfile:
    reader = csv.reader(csvFile, delimiter=',', quotechar='"')
    writer = csv.writer(tempfile, delimiter=',', quotechar='"')
    for row in reader:
        if row[4] != 'culture':
            row[4] = cultures.index(row[4])
        if row[5] != 'house':
            row[5] = houses.index(row[5])
        writer.writerow(row)

shutil.move(tempfile.name, filename)
