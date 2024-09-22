from matplotlib import pyplot
import pandas as pd
import os

pyplot.figure(figsize=[10, 7])
pyplot.xlabel("Bytes of data")
pyplot.ylabel("Seconds")
pyplot.title("Average communication time as a function of data size in bytes")
pyplot.grid(visible=True)

for file in os.listdir(r"./data"):
    s = file.split(sep='.')
    path = os.path.join(r"./outputs", s[0] + ".png")

    data = pd.read_csv(r"./data/" + file, header = None)

    x = data[0]; y = data[1]

    pyplot.plot(x,y, label=s[0])
    pyplot.scatter(x,y,marker="+")

pyplot.legend(loc = 'upper left')
pyplot.savefig(os.path.join(r"./outputs", "plots.png"))