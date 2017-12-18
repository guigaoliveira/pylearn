''' Algoritmo para plotar a série de fourier de
    f(x) = { 
        -1, -pi < x < 0
        1, 0 < x < pi 
    }
 '''
import numpy as np
import matplotlib.pyplot as plt


def fourier(i):
    x = np.linspace(-np.pi, np.pi, 1000)
    y = [0 for _ in x]
    print(type(y))
    for n in range(1, i):
        y += (4 / np.pi) * (np.sin(x * (2 * n - 1))) / (2 * n - 1)
    #plt.title('Aproximando com a série de Fourier')
    plt.plot(x, y)
    plt.grid(True)
    #plt.show()


plt.style.use('ggplot')
fig = plt.figure()
plt.title('Aproximando com a série de Fourier', fontsize=15)

fourier(5)
fourier(100)
plt.savefig("Fourier.png", bbox_inches='tight')