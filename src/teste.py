import pandas as pd
import numpy as np
from scipy.linalg import eig
from matplotlib import pyplot as plt
#############
# [14:17, 12/06/2023] +55 48 9179-2999: Dimensões do andar:                226x151,80x9,65mm
# [14:18, 12/06/2023] +55 48 9179-2999: Perfil da viga:                    49,66x3,2mm
# [14:23, 12/06/2023] +55 48 9179-2999: Distância entre vigas:             55mm
# [14:24, 12/06/2023] +55 48 9179-2999: Altura entre andares 0-1/1-2/2-3:  120mm / 90,03 mm / 90,67 mm
##############


## Sistemas Discretos - Sistema 3 GL
# Trabalho fundamentos de vibrações
# Simulação de sistema com vários graus de liberdade
######################

## Dados experimentais da FRF
df_frf = pd.read_csv("dados/FRF.csv") # Carregar dados experimentais


freq_exp = df_frf["freq (Hz)"] # vetor de frequência experimental

FRF_andar_1 = df_frf["Signal 2 (Real)"] + df_frf["Signal 2 (Imag.)"]*1j # FRF primeiro andar
FRF_andar_2 = df_frf["Signal 3 (Real)"] + df_frf["Signal 3 (Imag.)"]*1j # FRF segundo andar
FRF_andar_3 = df_frf["Signal 4 (Real)"] + df_frf["Signal 4 (Imag.)"]*1j # FRF terceiro andar

Aexp1 = FRF_andar_1 # Acelerância a ser analisada - selecionar
Aexp2 = FRF_andar_2 # Acelerância a ser analisada - selecionar
Aexp3 = FRF_andar_3 # Acelerância a ser analisada - selecionar

## Parâmetros iniciais

GL = 3 # número de graus de liberdade

espessura_placa_lateral = 3.2e-3                 # espessura da placa [m]
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
m_andar = rho*espessura_base*largura_base*comprimento_base # Massa de cada andar
# M = [m2 -0.4 m3]+m_andar         # Vetor de massa total por andar
# M = [m2-1.2 -1 m3-0.9]+m_andar   # Vetor de massa total por andar
M = np.array([m1, m2, 0]) + m_andar  # Vetor de massa total por andar

# Rigidez placa lateral

I = largura_placa_lateral*espessura_placa_lateral**3/12 # Momento de inércia

k_vigas = []
#k_vigas.append(4*3.8*E*I/(comprimento_placa[0]**3))  # rigidez equivalente 1 andar
#k_vigas.append(4*4.3*E*I/(comprimento_placa[1]**3))  # rigidez equivalente 2 andar
#k_vigas.append(4*10.6*E*I/(comprimento_placa[2]**3)) # rigidez equivalente 3 andar
k_vigas.append(4*3*E*I/(comprimento_placa[0]**3))  # rigidez equivalente 1 andar
k_vigas.append(4*3*E*I/(comprimento_placa[1]**3))  # rigidez equivalente 2 andar
k_vigas.append(4*3*E*I/(comprimento_placa[2]**3)) # rigidez equivalente 3 andar
# k_viga(1) = 4*3*E*I/(comprimento_placa(1).^3) # rigidez equivalente em cada andar
# k_viga(2) = 4*3*E*I/(comprimento_placa(2).^3) # rigidez equivalente em cada andar
# k_viga(3) = 4*3*E*I/(comprimento_placa(3).^3) # rigidez equivalente em cada andar



## Cálculo das matrizes Massa e Rigidez

M = np.eye(GL) * M               # Matriz diagonal de massas
print(M)
K = np.zeros((GL,GL))

K = np.array([
    [k_vigas[0]+k_vigas[1], -k_vigas[1],  0],
    [-k_vigas[1], k_vigas[1]+k_vigas[2], -k_vigas[2]],
    [0, -k_vigas[2], k_vigas[2]]
    ])

print(K)


# Extração dos autovalores e autovetores

W, V= eig(K, M)

fi = np.diag(np.sqrt(W)) / (2 * np.pi)

print("############## W ABAIXO")
print(W)
print("############## V ABAIXO")
print(V)
print("############## fi ABAIXO")
print(fi)


# plt.figure()
# grid on

# plot([0; V(:,1)],0:GL,'-ob','LineWidth',1.8);hold on
# plot([0; V(:,2)],0:GL,'-or','LineWidth',1.8);
# plot([0; V(:,3)],0:GL,'-ok','LineWidth',1.8);
# ylabel('Grau de Liberdade','Interpreter','latex'); 
# xlabel('Amplitude','Interpreter','latex');
# yticks(0:1:3)
# xticks(-0.7:0.1:0.7)
# set(gca,'FontSize',24); grid on
# legend('Primeiro modo de vibrar','Segundo modo de vibrar','Terceiro modo de vibrar' )


plt.figure(1)
plt.plot([0] + list(V[:, 0]), np.arange(GL+1), '-ob', linewidth=1.8)
plt.plot([0] + list(V[:, 1]), np.arange(GL+1), '-or', linewidth=1.8)
plt.plot([0] + list(V[:, 2]), np.arange(GL+1), '-ok', linewidth=1.8)
plt.xlabel('Amplitude')
plt.ylabel("Grau de Liberdade")
plt.grid(bool)
#plt.legend('Primeiro modo de vibrar','Segundo modo de vibrar','Terceiro modo de vibrar' )

#plt.savefig('Figura1.png')

# Amortecimento modal por meio da banda de meia potência

# ANDAR 1
fn_exp = []
# primeira ressonância
start_idx = 15*2
end_idx = 50*2
selected_values = Aexp1[start_idx:end_idx]
max_value = np.max(np.abs(selected_values))
cort_amplitude = max_value / np.sqrt(2)
idx = np.argmax(np.abs(selected_values))
fn_exp.append(freq_exp[start_idx+idx-1])
# criando linha horizontal
hp_1 = np.ones(len(Aexp1)) * cort_amplitude
f_corte_1 = np.array([19.75, 20.97]) # obtida pelo gráfico

# Segunda ressonância
start_idx = 65*2
end_idx = 80*2
selected_values = Aexp1[start_idx:end_idx]
max_value = np.max(np.abs(selected_values))
cort_amplitude = max_value / np.sqrt(2)
idx = np.argmax(np.abs(selected_values))
fn_exp.append(freq_exp[start_idx+idx-1])
# criando linha horizontal
hp_2 = np.ones(len(Aexp1)) * cort_amplitude
f_corte_2 = np.array([72.59 , 73.93]) # obtida pelo gráfico

# Terceira ressonância
start_idx = 120*2
end_idx = 150*2
selected_values = Aexp1[start_idx:end_idx]
max_value = np.max(np.abs(selected_values))
cort_amplitude = max_value / np.sqrt(2)
idx = np.argmax(np.abs(selected_values))
fn_exp.append(freq_exp[start_idx+idx-1])
# criando linha horizontal
hp_3 = np.ones((len(Aexp1),1)) * cort_amplitude
f_corte_3 = np.array([137.88 , 140.05]) # obtida pelo gráfico

## dados do gráfico meia potência - Amortecimento modal
zeta1 = []
zeta1.append((f_corte_1[1] - f_corte_1[0])/(2*fn_exp[0]))
zeta1.append((f_corte_2[1] - f_corte_2[0])/(2*fn_exp[1]))
zeta1.append((f_corte_3[1] - f_corte_3[0])/(2*fn_exp[2]))

## Amortecimento modal por meio da banda de meia potência
# ANDAR 2
fn_exp = []
# primeira ressonância
start_idx = 15*2
end_idx = 50*2
selected_values = Aexp2[start_idx:end_idx]
max_value = np.max(np.abs(selected_values))
cort_amplitude = max_value / np.sqrt(2)
idx = np.argmax(np.abs(selected_values))
fn_exp.append(freq_exp[start_idx+idx-1])
# criando linha horizontal
hp_1 = np.ones(len(Aexp2)) * cort_amplitude
f_corte_1 = np.array([19.8, 20.9]) # obtida pelo gráfico

# Segunda ressonância
start_idx = 65*2
end_idx = 80*2
selected_values = Aexp2[start_idx:end_idx]
max_value = np.max(np.abs(selected_values))
cort_amplitude = max_value / np.sqrt(2)
idx = np.argmax(np.abs(selected_values))
fn_exp.append(freq_exp[start_idx+idx-1])
# criando linha horizontal
hp_2 = np.ones(len(Aexp2)) * cort_amplitude
f_corte_2 = np.array([72.65 , 74.2]) # obtida pelo gráfico

# Terceira ressonância
start_idx = 120*2
end_idx = 150*2
selected_values = Aexp2[start_idx:end_idx]
max_value = np.max(np.abs(selected_values))
cort_amplitude = max_value / np.sqrt(2)
idx = np.argmax(np.abs(selected_values))
fn_exp.append(freq_exp[start_idx+idx-1])
# criando linha horizontal
hp_3 = np.ones((len(Aexp2),1)) * cort_amplitude
f_corte_3 = np.array([137.4 , 140]) # obtida pelo gráfico

zeta2 = []
zeta2.append((f_corte_1[1] - f_corte_1[0])/(2*fn_exp[0]))
zeta2.append((f_corte_2[1] - f_corte_2[0])/(2*fn_exp[1]))
zeta2.append((f_corte_3[1] - f_corte_3[0])/(2*fn_exp[2]))

## Amortecimento modal por meio da banda de meia potência
# ANDAR 3
fn_exp = []
# primeira ressonância
start_idx = 15*2
end_idx = 50*2
selected_values = Aexp3[start_idx:end_idx]
max_value = np.max(np.abs(selected_values))
cort_amplitude = max_value / np.sqrt(2)
idx = np.argmax(np.abs(selected_values))
fn_exp.append(freq_exp[start_idx+idx-1])
# criando linha horizontal
hp_1 = np.ones(len(Aexp3)) * cort_amplitude
f_corte_1 = np.array([19.83, 20.9]) # obtida pelo gráfico

# Segunda ressonância
start_idx = 65*2
end_idx = 80*2
selected_values = Aexp3[start_idx:end_idx]
max_value = np.max(np.abs(selected_values))
cort_amplitude = max_value / np.sqrt(2)
idx = np.argmax(np.abs(selected_values))
fn_exp.append(freq_exp[start_idx+idx-1])
# criando linha horizontal
hp_2 = np.ones(len(Aexp3)) * cort_amplitude
f_corte_2 = np.array([72.55, 73.9]) # obtida pelo gráfico

# Terceira ressonância
start_idx = 120*2
end_idx = 150*2
selected_values = Aexp3[start_idx:end_idx]
max_value = np.max(np.abs(selected_values))
cort_amplitude = max_value / np.sqrt(2)
idx = np.argmax(np.abs(selected_values))
fn_exp.append(freq_exp[start_idx+idx-1])
# criando linha horizontal
hp_3 = np.ones((len(Aexp3),1)) * cort_amplitude
f_corte_3 = np.array([137.3, 140]) # obtida pelo gráfico
# dados do gráfico meia potência - Amortecimento modal
zeta3 = []
zeta3.append((f_corte_1[1] - f_corte_1[0])/(2*fn_exp[0]))
zeta3.append((f_corte_2[1] - f_corte_2[0])/(2*fn_exp[1]))
zeta3.append((f_corte_3[1] - f_corte_3[0])/(2*fn_exp[2]))

# Funcoes Resposta em Frequencia acrescida de amortecimento
x = 3 # Grau de liberdade analisado
xf = 1 # Grau de liberdade com a excitação
wi = np.sqrt(W)
print("########### wi abaixo")
print(wi)
f = np.arange(0, 400.5, 0.5)
w = 2*np.pi*f

H1 = np.zeros(len(w),dtype=np.complex_)
print(H1)
S1 = np.zeros(len(w),dtype=np.complex_)
H3 = np.zeros(len(w),dtype=np.complex_)
print(len(w))
for n in range(len(w)):
    soma1=0
    soma2=0
    soma3=0
    for k in range(GL):

        soma1 += (V[0, k] * V[x-1, k]) / (wi[k]**2 - w[n]**2 + 1j * 2 * zeta1[k] * wi[k] * w[n])
        soma2 += (V[1, k] * V[x-1, k]) / (wi[k]**2 - w[n]**2 + 1j * 2 * zeta2[k] * wi[k] * w[n])
        soma3 += (V[2, k] * V[x-1, k]) / (wi[k]**2 - w[n]**2 + 1j * 2 * zeta3[k] * wi[k] * w[n])

    #TODO checar se tem imaginario tbm
    H1[n] = soma1
    S1[n] = soma2
    H3[n] = soma3

A1 = -w**2 * H1
A2 = -w**2 * S1
A3 = -w**2 * H3

## Comparação de Acelerâncias - excitação 1 e medida 1
plt.figure(2)
plt.semilogy(freq_exp,np.abs(Aexp1), color='blue')
plt.semilogy(f,abs(A1),'-r','LineWidth',2)
plt.xlim([0, 50*np.pi*2])
plt.xlabel("Frequência [Hz]")
plt.ylabel('Teste1')
# plt.grid(bool)

plt.savefig('Figura2.png')

## Comparação de Acelerâncias - excitação 1 e medida 2
plt.figure(3)
plt.semilogy(freq_exp,np.abs(Aexp2), color='blue')
plt.semilogy(f,abs(A2),'-r','LineWidth',2)
plt.xlim([0, 50*np.pi*2])
plt.xlabel("Frequência [Hz]")
plt.ylabel('Teste2')
# plt.grid(bool)

plt.savefig('Figura3.png')

## Comparação de Acelerâncias - excitação 1 e medida 3
plt.figure(4)
plt.semilogy(freq_exp,np.abs(Aexp3), color='blue')
plt.semilogy(f,abs(A3),'-r','LineWidth',2)
plt.xlim([0, 50*np.pi*2])
plt.xlabel("Frequência [Hz]")
plt.ylabel('Teste3')
# plt.grid(bool)

plt.savefig('Figura4.png')



## Dados experimentais do espectro
df_banda= pd.read_csv("dados/Passa-banda.csv.txt")
freq_exp = df_banda("passo")
Force = df_banda("Signal 1")
S_exp_1 = df_banda("Signal 2")
S_exp_2 = df_banda("Signal 3")
S_exp_3 = df_banda("Signal 4")

# Funcoes Resposta em Frequencia - Sem amortecimento

S1 = np.zeros(len(w),dtype=np.complex_)
S2 = np.zeros(len(w),dtype=np.complex_)
S3 = np.zeros(len(w),dtype=np.complex_)

for n in range(len(w)):
    soma1=0
    soma2=0
    soma3=0
    for k in range(GL):

        soma1 += (V[0, k] * V[x-1, k]) / (wi[k]**2 - w[n]**2 + 1j * 2 * zeta1[k] * wi[k] * w[n])
        soma2 += (V[1, k] * V[x-1, k]) / (wi[k]**2 - w[n]**2 + 1j * 2 * zeta2[k] * wi[k] * w[n])
        soma3 += (V[2, k] * V[x-1, k]) / (wi[k]**2 - w[n]**2 + 1j * 2 * zeta3[k] * wi[k] * w[n])

    S1[n] = soma1 * np.sqrt(Force[n])
    S2[n] = soma2 * np.sqrt(Force[n])
    S3[n] = soma3 * np.sqrt(Force[n])

SA1 = -w**2 * S1
SA2 = -w**2 * S2
SA3 = -w**2 * S3

# Espectro excitação 1 e medida em 1

plt.figure(5)
plt.plot(freq_exp,np.sqrt(S_exp_1),'k','linewidth',1.5)
plt.plot(f, abs(SA1),'r')
plt.xlim([50, 100])
#set(gca,'FontSize',24);
#grid on;
plt.xlabel('Frequência [Hz]')
plt.ylabel('Magnitude ($m/s^2$)')

plt.savefig("Figura5.png")
#legend('Resposta experimental','Resposta Simulado (FxA)','interpreter','latex')

## Espectro excitação 1 e medida em 2

plt.figure(6)
plt.plot(freq_exp,np.sqrt(S_exp_2),'k','linewidth',1.5)
plt.plot(f, abs(SA2),'r')
plt.xlim([50, 100])
#set(gca,'FontSize',24);
#grid on;
plt.xlabel('Frequência [Hz]')
plt.ylabel('Magnitude ($m/s^2$)')

plt.savefig("Figura6.png")

## Espectro excitação 1 e medida em 3

plt.figure(7)
plt.plot(freq_exp,np.sqrt(S_exp_3),'k','linewidth',1.5)
plt.plot(f, abs(SA3),'r')
plt.xlim([50, 100])
#set(gca,'FontSize',24);
#grid on;
plt.xlabel('Frequência [Hz]')
plt.ylabel('Magnitude ($m/s^2$)')

plt.savefig("Figura7.png")


## rms
S_exp_1_rms = S_exp_1/(np.sqrt(2))
S_exp_2_rms = S_exp_2/(np.sqrt(2))
S_exp_3_rms = S_exp_3/(np.sqrt(2))

S1_rms = S1/(np.sqrt(2))
S2_rms = S2/(np.sqrt(2))
S3_rms = S3/(np.sqrt(2))


##

plt.figure(8)
plt.subplot(1,3,1)
plt.plot(freq_exp,20*np.log10(S1_rms),'r','linewidth',2)
plt.plot(freq_exp,20*np.log10(S_exp_1_rms),'k','linewidth',2)
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequência [Hz]')
plt.xlim([0, 200])

plt.subplot(1,3,2)
plt.plot(freq_exp,20*np.log10(S2_rms),'r','linewidth',2)
plt.plot(freq_exp,20*np.log10(S_exp_2_rms),'k','linewidth',2)
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequência [Hz]')
plt.xlim([0, 200])


plt.subplot(1,3,3)
plt.plot(freq_exp,20*np.log10(S3_rms),'r','linewidth',2)
plt.plot(freq_exp,20*np.log10(S_exp_3_rms),'k','linewidth',2)
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequência [Hz]')
plt.xlim([0, 200])

plt.legend('Resposta Simulado (FxA)','Resposta experimental','interpreter','latex','FontSize',12)

plt.savefig("Figura8.png")
