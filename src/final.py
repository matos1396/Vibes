import pandas as pd
import numpy as np
from scipy.linalg import eig
from matplotlib import pyplot as plt
import matplotlib

# Configuração
matplotlib.rcParams["figure.dpi"]= 600

## Sistemas Discretos - Sistema 3 GL
# Trabalho fundamentos de vibrações
# Simulação de sistema com vários graus de liberdade
######################

## Dados experimentais da FRF
df_frf = pd.read_csv("dados/FRF.csv") # Carregar dados experimentais
freq_exp = df_frf["freq (Hz)"] # vetor de frequência experimental

FRF_andar_1 = df_frf["Signal 2 (Real)"] + df_frf["Signal 2 (Imag.)"]*1j # Dados FRF primeiro andar
FRF_andar_2 = df_frf["Signal 3 (Real)"] + df_frf["Signal 3 (Imag.)"]*1j # Dados FRF segundo andar
FRF_andar_3 = df_frf["Signal 4 (Real)"] + df_frf["Signal 4 (Imag.)"]*1j # Dados FRF terceiro andar

## Parâmetros iniciais
GL = 3 # número de graus de liberdade

espessura_placa_lateral = 3.2e-3                  # espessura da placa [m]
largura_placa_lateral = 49.66e-3                  # largura da placa [m]
comprimento_placa = [120e-3, 90.03e-3, 90.67e-3] # comprimento da placa [m]

# Dimensões e propriedades da base     226x151,80x9,65mm
espessura_base = 9.65e-3      # espessura da base [m]
comprimento_base = 226e-3     # comprimento da base [m]
largura_base = 151.8e-3       # largura da base [m]
E = 205e+9                    # módulo de elasticidade do aço
rho = 7589                    # densidade do aço

# Massa no andar
m1 = 0.646 # Massa adicional [kg]
m2 = 1.187 # Massa adicional [kg]
m3 = 0
m_andar = rho*espessura_base*largura_base*comprimento_base # Massa de cada andar
M = np.array([m1, m2, m3]) + m_andar  # Vetor de massa total por andar

# Rigidez placa lateral
I = largura_placa_lateral*espessura_placa_lateral**3/12 # Momento de inércia

k_vigas = []
# k_vigas.append(3*E*I/(comprimento_placa[0]**3))             # rigidez equivalente 1 andar - Viga em Balanço
# k_vigas.append(3*E*I/(comprimento_placa[1]**3))             # rigidez equivalente 2 andar - Viga em Balanço
# k_vigas.append(3*E*I/(comprimento_placa[2]**3))             # rigidez equivalente 3 andar - Viga em Balanço
k_vigas.append(12*E*I/(comprimento_placa[0]**3))             # rigidez equivalente 1 andar - Sem Ajuste
k_vigas.append(12*E*I/(comprimento_placa[1]**3))             # rigidez equivalente 2 andar - Sem Ajuste
k_vigas.append(12*E*I/(comprimento_placa[2]**3))             # rigidez equivalente 3 andar - Sem Ajuste
# k_vigas.append(15/(3.5*2)*E*I/(comprimento_placa[0]**3))        # rigidez equivalente 1 andar - Com Ajuste
# k_vigas.append(15/(2.5*2)*E*I/(comprimento_placa[1]**3))        # rigidez equivalente 2 andar - Com Ajuste
# k_vigas.append(15/(2)*E*I/(comprimento_placa[2]**3))            # rigidez equivalente 3 andar - Com Ajuste

# k_vigas.append(15*10/(2)*E*I/(comprimento_placa[2]**3))       # rigidez equivalente 3 andar - Para o Item 7 - (Reduzir Vibrações no Andar 3)

# Como tem 4 vigas, o k do andar será multiplicado por 4
k_andar = []
for k in range(len(k_vigas)):
    k_vigas[k] *= 4
    k_andar.append(k_vigas[k])


## Cálculo das matrizes Massa e Rigidez

M = np.eye(GL) * M               # Matriz diagonal de massas
K = np.zeros((GL,GL))

K = np.array([
    [k_andar[0]+k_andar[1], -k_andar[1],  0],
    [-k_andar[1], k_andar[1]+k_andar[2], -k_andar[2]],
    [0, -k_andar[2], k_andar[2]]
    ])

# Extração dos autovalores e autovetores
W, V= eig(K, M)
fi = np.diag(np.sqrt(W)) / (2 * np.pi)

# Modos de vibração
plt.figure(1)
plt.plot(np.arange(GL+1), [0] + list(V[:, 0]), "-ob", linewidth=1.8, label = "1º Modo")
plt.plot(np.arange(GL+1), [0] + list(V[:, 1]), "-or", linewidth=1.8, label = "2º Modo")
plt.plot(np.arange(GL+1), [0] + list(V[:, 2]), "-ok", linewidth=1.8, label = "3º Modo")
plt.ylabel("Amplitude de Vibração")
plt.xlabel("Andar")
plt.xticks(np.arange(0, 4, 1))
plt.xlim(0,3)
plt.legend()
plt.grid(True, alpha = 0.5)

plt.savefig("formas_modais.png")

# Amortecimento modal por meio da banda de meia potência
# ANDAR 1

fn_exp = []

# Primeiro Pico
start_idx = 15*2  
end_idx = 50*2    
selected_values = FRF_andar_1[start_idx:end_idx]
max_value = np.max(np.abs(selected_values))
cort_amplitude = max_value / np.sqrt(2)
idx = np.argmax(np.abs(selected_values))
fn_exp.append(freq_exp[start_idx+idx])

# Corte
f_corte_1 = np.array([20.23, 21.92]) # obtida pelo gráfico

# Gráfico
plt.figure(10, figsize=(20, 18))

plt.semilogy(freq_exp, np.abs(FRF_andar_1), color="blue")
plt.axhline(cort_amplitude, color="green", linestyle="-")
plt.xlim([start_idx/4, end_idx/4])
plt.xticks(np.arange(start_idx/2, end_idx/2, 0.25))
plt.xticks(rotation=90)
plt.xlabel("Frequência [Hz]")

# Segundo Pico
start_idx = 60*2
end_idx = 100*2
selected_values = FRF_andar_1[start_idx:end_idx]
max_value = np.max(np.abs(selected_values))
cort_amplitude = max_value / np.sqrt(2)
idx = np.argmax(np.abs(selected_values))
fn_exp.append(freq_exp[start_idx+idx])

# Gráfico
plt.figure(11, figsize=(20, 18))

plt.semilogy(freq_exp, np.abs(FRF_andar_1), color="blue")
plt.axhline(cort_amplitude, color="green", linestyle="-")
plt.xlim([start_idx/2, end_idx/2])
plt.xticks(np.arange(start_idx/2, end_idx/2, 0.5))
plt.xticks(rotation=90)
plt.xlabel("Frequência [Hz]")

# Corte
f_corte_2 = np.array([76.50 , 77.60]) # obtida pelo gráfico

# Terceiro Pico
start_idx = 120*2
end_idx = 200*2
selected_values = FRF_andar_1[start_idx:end_idx]
max_value = np.max(np.abs(selected_values))
cort_amplitude = max_value / np.sqrt(2)
idx = np.argmax(np.abs(selected_values))
fn_exp.append(freq_exp[start_idx+idx])


# Gráfico
plt.figure(12, figsize=(20, 18))
plt.semilogy(freq_exp, np.abs(FRF_andar_1), color="blue")
plt.axhline(cort_amplitude, color="green", linestyle="-")
plt.xlim([start_idx/2, end_idx/2])
plt.xticks(np.arange(start_idx/2, end_idx/2, 0.5))
plt.xticks(rotation=90)

plt.xlabel("Frequência [Hz]")
#plt.savefig("TESTE13.png")

# criando linha horizontal
f_corte_3 = np.array([143.16 , 146.45]) # obtida pelo gráfico


## dados do gráfico meia potência - Amortecimento modal
csi1 = []
csi1.append((f_corte_1[1] - f_corte_1[0])/(2*fn_exp[0]))
csi1.append((f_corte_2[1] - f_corte_2[0])/(2*fn_exp[1]))
csi1.append((f_corte_3[1] - f_corte_3[0])/(2*fn_exp[2]))

# Amortecimento Estrutural
eta1 = 2*csi1


##### ANDAR 2

fn_exp = []

# Primeiro Pico
start_idx = 15*2
end_idx = 50*2
selected_values = FRF_andar_2[start_idx:end_idx]
max_value = np.max(np.abs(selected_values))
cort_amplitude = max_value / np.sqrt(2)
idx = np.argmax(np.abs(selected_values))
fn_exp.append(freq_exp[start_idx+idx])

# Gráfico
plt.figure(13, figsize=(20, 18))

plt.semilogy(freq_exp, np.abs(FRF_andar_2), color="blue")
plt.axhline(cort_amplitude, color="green", linestyle="-")
plt.xlim([start_idx/2, end_idx/2])
plt.xticks(np.arange(start_idx/2, end_idx/2, 0.25))
plt.xticks(rotation=90)
plt.xlabel("Frequência [Hz]")

# Corte
f_corte_1 = np.array([20.29, 21.85]) # obtida pelo gráfico

# Segundo Pico
start_idx = 60*2
end_idx = 80*2
selected_values = FRF_andar_2[start_idx:end_idx]
max_value = np.max(np.abs(selected_values))
cort_amplitude = max_value / np.sqrt(2)
idx = np.argmax(np.abs(selected_values))
fn_exp.append(freq_exp[start_idx+idx])

# Gráfico
plt.figure(14, figsize=(20, 18))

plt.semilogy(freq_exp, np.abs(FRF_andar_2), color="blue")
plt.axhline(cort_amplitude, color="green", linestyle="-")
plt.xlim([start_idx/2, end_idx/2])
plt.xticks(np.arange(start_idx/2, end_idx/2, 0.25))
plt.xticks(rotation=90)
plt.xlabel("Frequência [Hz]")

# Corte
f_corte_2 = np.array([76.60 , 77.64]) # obtida pelo gráfico

# Terceiro Pico
start_idx = 120*2
end_idx = 150*2
selected_values = FRF_andar_3[start_idx:end_idx]
max_value = np.max(np.abs(selected_values))
cort_amplitude = max_value / np.sqrt(2)
idx = np.argmax(np.abs(selected_values))
fn_exp.append(freq_exp[start_idx+idx])

# Gráfico
plt.figure(15, figsize=(20, 18))

plt.semilogy(freq_exp, np.abs(FRF_andar_3), color="blue")
plt.axhline(cort_amplitude, color="green", linestyle="-")
plt.xlim([start_idx/2, end_idx/2])
plt.xticks(np.arange(start_idx/2, end_idx/2, 0.25))
plt.xticks(rotation=90)
plt.xlabel("Frequência [Hz]")

# Corte
f_corte_3 = np.array([142.98, 146.10]) # obtida pelo gráfico

## Amortecimento modal por meio da banda de meia potência
csi2 = []
csi2.append((f_corte_1[1] - f_corte_1[0])/(2*fn_exp[0]))
csi2.append((f_corte_2[1] - f_corte_2[0])/(2*fn_exp[1]))
csi2.append((f_corte_3[1] - f_corte_3[0])/(2*fn_exp[2]))

# Amortecimento Estrutural
eta2 = 2*csi2


# ANDAR 3
fn_exp = []

# Primeiro Pico
start_idx = 15*2
end_idx = 50*2
selected_values = FRF_andar_3[start_idx:end_idx]
max_value = np.max(np.abs(selected_values))
cort_amplitude = max_value / np.sqrt(2)
idx = np.argmax(np.abs(selected_values))
fn_exp.append(freq_exp[start_idx+idx])

plt.figure(figsize=(20, 18))

plt.semilogy(freq_exp, np.abs(FRF_andar_3), color="blue")
plt.axhline(cort_amplitude, color="green", linestyle="-")
plt.xlim([start_idx/2, end_idx/2])
plt.xticks(np.arange(start_idx/2, end_idx/2, 0.25))
plt.xticks(rotation=90)
plt.xlabel("Frequência [Hz]")
plt.savefig("TESTE31.png")

# Corte
f_corte_1 = np.array([20.28, 21.85]) # obtida pelo gráfico

# Segundo Pico
start_idx = 65*2
end_idx = 80*2
selected_values = FRF_andar_3[start_idx:end_idx]
max_value = np.max(np.abs(selected_values))
cort_amplitude = max_value / np.sqrt(2)
idx = np.argmax(np.abs(selected_values))
fn_exp.append(freq_exp[start_idx+idx-1])

plt.figure(16, figsize=(20, 18))

plt.semilogy(freq_exp, np.abs(FRF_andar_3), color="blue")
plt.axhline(cort_amplitude, color="green", linestyle="-")
plt.xlim([start_idx/2, end_idx/2])
plt.xticks(np.arange(start_idx/2, end_idx/2, 0.25))
plt.xticks(rotation=90)
plt.xlabel("Frequência [Hz]")
plt.savefig("TESTE32.png")

# Corte
f_corte_2 = np.array([76.51, 77.63]) # obtida pelo gráfico

# Terceiro Pico
start_idx = 120*2
end_idx = 150*2
selected_values = FRF_andar_3[start_idx:end_idx]
max_value = np.max(np.abs(selected_values))
cort_amplitude = max_value / np.sqrt(2)
idx = np.argmax(np.abs(selected_values))
fn_exp.append(freq_exp[start_idx+idx-1])

plt.figure(17, figsize=(20, 18))

plt.semilogy(freq_exp, np.abs(FRF_andar_3), color="blue")
plt.axhline(cort_amplitude, color="green", linestyle="-")
plt.xlim([start_idx/2, end_idx/2])
plt.xticks(np.arange(start_idx/2, end_idx/2, 0.25))
plt.xticks(rotation=90)
plt.xlabel("Frequência [Hz]")
plt.savefig("TESTE33.png")

# Corte
f_corte_3 = np.array([142.9, 146.1]) # obtida pelo gráfico


# dados do gráfico meia potência - Amortecimento modal
csi3 = []
csi3.append((f_corte_1[1] - f_corte_1[0])/(2*fn_exp[0]))
csi3.append((f_corte_2[1] - f_corte_2[0])/(2*fn_exp[1]))
csi3.append((f_corte_3[1] - f_corte_3[0])/(2*fn_exp[2]))

# Amortecimento Estrutural
eta3 = 2*csi3


# Funcoes Resposta em Frequencia acrescida de amortecimento
x = 3 # Grau de liberdade analisado
xf = 1 # Grau de liberdade com a excitação
wi = np.sqrt(W)
f = np.arange(0, 400.5, 0.5)
w = 2*np.pi*f

H1 = np.zeros(len(w), dtype=np.complex_)
S1 = np.zeros(len(w), dtype=np.complex_)
H3 = np.zeros(len(w), dtype=np.complex_)

for n in range(len(w)):
    soma1=0
    soma2=0
    soma3=0
    for k in range(GL):

        soma1 += (V[0, k] * V[x-1, k]) / (wi[k]**2 - w[n]**2 + 1j * eta1[k] * wi[k] * w[n])
        soma2 += (V[1, k] * V[x-1, k]) / (wi[k]**2 - w[n]**2 + 1j * eta2[k] * wi[k] * w[n])
        soma3 += (V[2, k] * V[x-1, k]) / (wi[k]**2 - w[n]**2 + 1j * eta3[k] * wi[k] * w[n])

    H1[n] = soma1
    S1[n] = soma2
    H3[n] = soma3

A1 = -w**2 * H1
A2 = -w**2 * S1
A3 = -w**2 * H3

## Comparação de Acelerâncias - Andar 1
plt.figure(2)
plt.semilogy(freq_exp, np.abs(FRF_andar_1), color="blue", label = "Experimental")
plt.semilogy(f, abs(A1),"-r", label = "Análitico (Estrutural)")
plt.xlim([0, 300])
plt.xlabel("Frequência [Hz]")
plt.ylabel("Acelerância [(m/s²)/N]")
plt.legend()
plt.grid(True, alpha = 0.5)

plt.savefig("Acelerância Andar - 1.png")

## Comparação de Acelerâncias - Andar 2
plt.figure(3)
plt.semilogy(freq_exp, np.abs(FRF_andar_2), color = "blue", label = "Experimental")
plt.semilogy(f,abs(A2),"-r", label = "Análitico (Estrutural)")
plt.xlim([0, 300])
plt.xlabel("Frequência [Hz]")
plt.ylabel("Acelerância [(m/s²)/N]")
plt.legend()
plt.grid(True, alpha = 0.5)

plt.savefig("Acelerância Andar - 2.png")

## Comparação de Acelerâncias - Andar 3

plt.figure(4)
plt.semilogy(freq_exp,np.abs(FRF_andar_3), color="blue", label = "Experimental")
plt.semilogy(f,abs(A3),"-r", label = "Análitico (Estrutural)")
plt.xlim([0, 300])
plt.xlabel("Frequência [Hz]")
plt.ylabel("Acelerância [(m/s²)/N]")
plt.legend()
plt.grid(True, alpha = 0.5)

plt.savefig("Acelerância Andar - 3.png")

## Dados experimentais do espectro
df_banda= pd.read_csv("dados/Passa-banda.csv")
freq_exp = df_banda["passo"]
Force = df_banda["Signal 1"]
S_exp_1 = df_banda["Signal 2"]
S_exp_2 = df_banda["Signal 3"]
S_exp_3 = df_banda["Signal 4"]


# Funcoes Resposta em Frequencia - Sem amortecimento
S1 = np.zeros(len(w), dtype=np.complex_)
S2 = np.zeros(len(w), dtype=np.complex_)
S3 = np.zeros(len(w), dtype=np.complex_)

for n in range(len(w)):
    soma1=0
    soma2=0
    soma3=0
    for k in range(GL):

        soma1 += (V[0, k] * V[x-1, k]) / (wi[k]**2 - w[n]**2 + 1j * eta1[k] * wi[k] * w[n])
        soma2 += (V[1, k] * V[x-1, k]) / (wi[k]**2 - w[n]**2 + 1j * eta2[k] * wi[k] * w[n])
        soma3 += (V[2, k] * V[x-1, k]) / (wi[k]**2 - w[n]**2 + 1j * eta3[k] * wi[k] * w[n])

    S1[n] = soma1 * np.sqrt(Force[n])
    S2[n] = soma2 * np.sqrt(Force[n])
    S3[n] = soma3 * np.sqrt(Force[n])

SA1 = -w**2 * S1
SA2 = -w**2 * S2
SA3 = -w**2 * S3

# Espectro excitação Andar 1
plt.figure(5)
plt.plot(freq_exp,np.sqrt(S_exp_1), label = "Resposta Experimental" )
plt.plot(f, abs(SA1), label = "Resposta Analítica")
plt.xlim([50, 100])
plt.xticks(np.arange(50, 101, 10))
plt.grid(True, alpha = 0.5)
plt.legend()
plt.xlabel("Frequência [Hz]")
plt.ylabel("Magnitude ($m/s^2$)")

plt.savefig("magnitude_50-100_A1.png")

# Espectro excitação Andar 2
plt.figure(6)
plt.plot(freq_exp,np.sqrt(S_exp_2), label = "Resposta Experimental")
plt.plot(f, abs(SA2), label = "Resposta Analítica")
plt.xlim([50, 100])
plt.xticks(np.arange(50, 101, 10))
plt.grid(True, alpha = 0.5)
plt.legend()
plt.xlabel("Frequência [Hz]")
plt.ylabel("Magnitude ($m/s^2$)")

plt.savefig("magnitude_50-100_A2.png")

# Espectro excitação Andar 3
plt.figure(7)
plt.plot(freq_exp,np.sqrt(S_exp_3), label = "Resposta Experimental")
plt.plot(f, abs(SA3), label = "Resposta Analítica")
plt.xlim([50, 100])
plt.xticks(np.arange(50, 101, 10))
plt.grid(True, alpha = 0.5)
plt.legend()
plt.xlabel("Frequência [Hz]")
plt.ylabel("Magnitude ($m/s^2$)")

plt.savefig("magnitude_50-100_A3.png")

## RMS
S_exp_1_rms = S_exp_1/(np.sqrt(2))
S_exp_2_rms = S_exp_2/(np.sqrt(2))
S_exp_3_rms = S_exp_3/(np.sqrt(2))

S1_rms = S1/(np.sqrt(2))
S2_rms = S2/(np.sqrt(2))
S3_rms = S3/(np.sqrt(2))

## Gráficos RMS
plt.figure(20)
plt.plot(freq_exp,20*np.log10(S1_rms), label = "Simulado")
plt.plot(freq_exp,20*np.log10(S_exp_1_rms), label = "Experimental")
plt.grid(True, alpha = 0.5)
plt.legend()
plt.ylabel("Magnitude (dB) [dB ref. 1 m/N]")
plt.xlabel("Frequência [Hz]")
plt.xlim([0, 200])
plt.savefig("rms1.png")

plt.figure(21)
plt.plot(freq_exp,20*np.log10(S2_rms), label = "Simulado")
plt.plot(freq_exp,20*np.log10(S_exp_2_rms), label = "Experimental")
plt.grid(True, alpha = 0.5)
plt.legend()
plt.ylabel("Magnitude (dB) [dB ref. 1 m/N]")
plt.xlabel("Frequência [Hz]")
plt.xlim([0, 200])
plt.savefig("rms2.png")

plt.figure(22)
plt.plot(freq_exp,20*np.log10(S3_rms), label = "Simulado")
plt.plot(freq_exp,20*np.log10(S_exp_3_rms), label = "Experimental")
plt.grid(True, alpha = 0.5)
plt.legend()
plt.ylabel("Magnitude (dB) [dB ref. 1 m/N]")
plt.xlabel("Frequência [Hz]")
plt.xlim([0, 200])
plt.savefig("rms3.png")
