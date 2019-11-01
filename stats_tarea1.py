#Importar librerias
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from datetime import datetime
import numpy as np



#Leer csv y leer fechas correctamente ^^
df = pd.read_csv("homicidios.csv", parse_dates=['fecha'], infer_datetime_format = True)

#Veremos el head del dataframe:
print(df.head())

#Transformar datos
###listo, en la misma lectura de csv se transformaron las fechas

#Veamos nuestra gráfica original
###para ellos la segunda libreria

def plot_df(df, x, y, title="", xlabel='Date', ylabel='# de homicidios', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:blue')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    #plt.show()

plot_df(df, x=df.fecha, y=df.homicidios, title='Homicidios desde 2007')   

#Modelar
### (1,1,2) ARIMA model
#### tercer librería
model = ARIMA(df.fecha, order=(1,1,2))
model_fit = model.fit(disp=0)
#print(model_fit.summary())

#residuos
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
#plt.show()

#Elegir fecha de inicio y fin 

#### original vs fitted
#DUDA : en esta parte me marcó error con las fechas...?
#DUDA : cómo elijo la fecha de inicio y fin?
#forecast = model_fit.predict(start = df.fecha[0], 
#	end = df.fecha[len(df.fecha)], exog = None, 
#	typ = 'levels', dynamic = False)



model_fit.plot_predict(start = datetime.datetime('2007-01-01 00:00:00'), 
	end = datetime.datetime('2007-01-01 00:00:00'),
	dynamic = False)
#plt.show()

#Obtener gráfica con proyección




