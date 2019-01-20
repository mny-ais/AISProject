import matplotlib.pyplot as plt


class PlotIt():
    """ Generates plot of loss function from txt file """

    def __init__(self):
        file_name = "/home/aisgrp3/Documents/src_ln/plotdata.txt" 
        
        with open(file_name, "r") as file:
            data = file.read()

        open(file_name, "w").close()
        xaxis = []
        print("yay im here")
        data = data.splitlines()
        print(data)
        for i in range(0,len(data)):
            data[i] = float(data[i])
            xaxis.append(i)
        plt.bar(xaxis, data, 1/1.5, color="blue")
        plt.show(block=True)


if __name__ == "__main__":
        print("hi im nina")
        PlotIt()
