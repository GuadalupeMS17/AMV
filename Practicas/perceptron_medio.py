import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.linear_model import Perceptron


# Generación de datos sinteticos
X,y = make_blobs(100,2, centers=2, cluster_std=0.2)

# Conjunto de entrenamiento
Sn = pd.DataFrame(X,columns=['x1','x2'])
Sn['y']=y
Sn.head()


sns.lmplot(data=Sn, x='x1',y='x2',hue='y', fit_reg=False)

# Implementación del Perceptrón medio
def PerceptronMedio (Sn:pd.DataFrame,T:int)->tuple:
    k=0  #Esto es solo para contar los errores
    d= Sn.shape[1]-1
    n= Sn.shape[0]
    theta= np.zeros(d)
    thetagorro =np.zeros(d)
    theta0= 0.0
    theta0gorro = 0.0
    c = 1

    for t in range(T):
        for i in range(n):
            x=Sn.loc[i].values[:d]
            y=Sn.loc[i].values[-1]
            if y*(np.sign(np.dot(theta,x+theta0)))<=0:
                k += 1
                theta += y*x
                theta0 +=y   
                thetagorro += c*y*x
                theta0gorro +=c*y
                # print(theta,theta0)
            c -=1/(n*T)
    return thetagorro,theta0gorro,k

thetagorro,theta0gorro,k=PerceptronMedio(Sn,10)
thetagorro,theta0gorro,k

# Ecuación del Clasificador
sns.lmplot(data=Sn, x='x1',y='x2', hue='y', fit_reg=False)
x1 = np.linspace(-10,10,100)
x2 = (-theta0gorro-thetagorro[0]*x1)/thetagorro[1]
plt.plot(x1,x2,linestyle='--', color='blue')
plt.show()


