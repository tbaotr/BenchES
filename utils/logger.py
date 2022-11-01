# Adapted from:
# https://github.com/google-research/google-research/blob/master/persistent_es/logger.py
import csv


class CSVLogger():

    def __init__(self, fieldnames, filename='log.csv'):

        self.csv_file = open(filename, 'w')
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()
        self.csv_file.flush()

    def writerow(self, row):

        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):

        self.csv_file.close()
