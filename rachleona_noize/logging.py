import csv

class Logger:
    def __init__(self, *args):
        self._logs = {}
        for i in args:
            self._logs[i] = []

    def log(self, key, value):
        self._logs[key].append(value)

    def save(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for k in self._logs:
                writer.writerow([k] + self._logs[k])