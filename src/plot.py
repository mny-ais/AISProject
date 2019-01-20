import matplotlib.pyplot as plt


class PlotIt():
    """ Generates plot of loss function from txt file """

    def __init__(self):
        with open("/home/aisgrp3/Documents/src_ln/plotdata.txt", "r") as file:
            data = file.read()

        xaxis = []
        print("yay im here")
        data = data.splitlines()
        print(data)
        for i in range(0,len(data)):
            data[i] = float(data[i])
            xaxis.append(i)
        plt.bar(len(data), data, 1/1.5, color="blue")
        plt.show(block=True)


if __name__ == "__main__":
        print("hi im nina")
        PlotIt()
