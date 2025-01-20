import csv

class Logger:
    def __init__(self, *args):
        self.__logs = {}
        for i in args:
            self.__logs[i] = []

    def log(self, key, value):
        self.__logs[key].append(value)

    def save(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for k in self.__logs:
                writer.writerow([k] + self._logs[k])