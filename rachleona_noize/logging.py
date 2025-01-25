import csv
import torch


class Logger:
    """
    Logger class for logging loss terms during perturbation generation

    ...

    Attributes
    ----------
    __logs: dict
        private dictionary of lists containing logged values
        keys are set at initialisation

    Methods
    -------
    log(key, value)
        adds a value to the list of given key

    save(filename)
        save current log into a csv file of the given filename
    """

    def __init__(self, *args):
        self.__logs = {}
        for i in args:
            self.__logs[i] = []

    def log(self, key, value):
        """
        adds a value to the list of given key

        Parameters
        ----------
        key : str
            source audio tensor to compare to
        value : any
            value to be logged
        """
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.__logs[key].append(value)

    def save(self, filename):
        """
        save current log into a csv file of the given filename
        each key will have its own row, while each column is one iteration

        Parameters
        ----------
        filename : str
            name of the file to save logs to
        """
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for k in self.__logs:
                writer.writerow([k] + self.__logs[k])
