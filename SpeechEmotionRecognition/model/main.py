import numpy as np
import pandas as pd
from scipy import signal #https://docs.scipy.org/doc/scipy-0.18.1/reference/signal.html
import wave
# criar um fluxo de projeto onde outras classes são chamadas dentro desta main e que elas poderão ser testadas ou trocadas depois

# Leitura dos dados de entrada

# referencia dos datasets:
# https://www.reddit.com/r/datasets/comments/74f4hz/emotional_speech_datasets/
# database em portugues (VERBO) - https://thescipub.com/abstract/jcssp.2018.1420.1430
# https://sites.google.com/view/verbodatabase/home/download tem que entrar aqui, imprimir e assinar um form
# SAVEE-english-uk database: http://kahlan.eps.surrey.ac.uk/savee/Data/  -> username: "guest2savee" and the following password: "welcome!"
data = wave.open('../databases/SAVEE-english-uk/AudioData/DC/a01.wav')

# Tratamento dos dados de entrada: extração de caracteristicas do arquivo de audio

# Treinamento do modelo

# Teste do modelo

# Deploy?

