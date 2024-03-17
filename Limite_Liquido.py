import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

suelo_humedo_deposito = np.array([17.65, 18.87, 20.59])
suelo_seco_deposito = np.array([14.73, 15.63, 16.95])
peso_deposito = np.array([2.63, 2.74, 2.96])
numero_golpes = np.array([28, 23, 18])

#Calculando
peso_agua = np.subtract(suelo_humedo_deposito,suelo_seco_deposito )
suelo_seco = np.subtract(suelo_seco_deposito, peso_deposito )
contenido_humedad = np.divide(peso_agua, suelo_seco)
contenido_humedad *=100

#Creandando nuestra dataframe

df = pd.DataFrame({'NumeroGolpes': numero_golpes, 'ContenidoHumedad': contenido_humedad})
df['LogNumeroGolpes']= np.log(df['NumeroGolpes'])

#Ajuste de regresion lineal
modelo = LinearRegression()
x = df['LogNumeroGolpes'].values.reshape(-1,1)
y = df['ContenidoHumedad']
modelo.fit(x,y)

#funcion de prediccion
prediccion = modelo.predict(df[['LogNumeroGolpes']])

#Calcular el limite liquido
limite_liquido = modelo.predict([[np.log(25)]])[0]

#Construir el grafico con matplotlib
plt.scatter(np.log(df['NumeroGolpes']), df['ContenidoHumedad'], color='green')
plt.plot(np.log(df[['NumeroGolpes']]), prediccion)
plt.title('CONTENIDO DE HUMEDAD A 25 GOLPES')
plt.xlabel('NUMERO DE GOLPES')
plt.ylabel('CONTENIDO DE HUMEDAD(%)')

#Calculando el punto de interseccion para 25 golpes
x_interseccion = np.log(25)
y_interseccion = modelo.predict(np.log([[25]]))[0]
plt.scatter(x_interseccion, y_interseccion, color='red')

# Estableciendo valores en el eje X
valores_golpes = [10, 15, 20, 25, 30, 40, 50, 60]
valore_golpes_log = np.log(valores_golpes)
plt.xticks(valore_golpes_log ,valores_golpes )

xmax= np.log(90)
ymax= 30

plt.xlim(0, xmax)
plt.ylim(20, ymax)

plt.plot([x_interseccion, x_interseccion], [0,y_interseccion], color ='red', linestyle='--')
plt.plot([0, x_interseccion], [y_interseccion,y_interseccion], color ='red', linestyle='--')

for golpe in [15,20,30,40,50,60]:
    plt.axvline(x=np.log(golpe), color ='gray', linestyle='--')

for humedad in [20,22,24,26,28]:
    plt.axhline(humedad, color ='gray', linestyle='--')

legenda = f'Limite Liquido(%)={limite_liquido:.2f}'
plt.legend([legenda])


plt.show()