import matplotlib.pyplot as plt


class PlotIt(file):
    """ Generates plot of loss function from txt file """

    with open(file, "r") as file:
        data = file.read()

    xaxis = []

    data = data.splitlines()
    for i in range(0,len(data)):
        data[i] = int(data[i])
        xaxis.append(i)

    plt.bar(xaxis, data)
    plt.show()
    

    
 

    
