from matplotlib import pyplot
import csv
import pandas as pd
import numpy as np

def plot_measurements(filename, output):
    data = pd.read_csv(filename, header = None)
    x = data[0]; y = data[1]

    pyplot.plot(x,y)
    pyplot.scatter(x,y,marker="+")
    pyplot.xlabel("Bytes of data")
    pyplot.ylabel("Seconds")
    pyplot.lab(filename)
    pyplot.grid(visible=True)
