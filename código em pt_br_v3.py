import numpy as np
import pandas as pd
from scipy.linalg import eig
from scipy.signal import find_peaks
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.dpi']=600 # config



dados_df = pd.read_csv("dados/FRF.csv", sep = ",")

# Parte 1 - Configuração do programa
np.set_printoptions(precision=3, suppress=True)
#i = sqrt(-1)  # unidade complexa
g = 9.81  # aceleração da gravidade, em m/s^2

# Modelo Analítico
freqMaxAna = 40  # frequência máxima solicitada no trabalho, em Hz
resAna = 0.005  # resolução do modelo analítico. Você tem que definir um valor pequeno para que a discretização se aproxime bem do modelo contínuo
vetFreqAna = np.arange(0, freqMaxAna + resAna, resAna)  # vetor de frequência (para plotar no gráfico mais tarde)
wAna = 2 * np.pi * vetFreqAna  # vetor de frequência em rad/s
#print(wAna)
# Modelo Experimental
freqMaxExp = 40  # frequência máxima solicitada no trabalho, em Hz
resFreq = 0.5  # resolução de frequência do acelerômetro, em Hz
nPontosExp = int(freqMaxExp / resFreq + 1)  # número de pontos do modelo experimental (tem este +1 para incluir os 2 extremos)
vetFreqExp = np.linspace(0, freqMaxExp, nPontosExp)  # vetor de frequência (para plotar no gráfico mais tarde)
# vetFreqAna = np.arange(0, freqMaxAna + resAna + resAna, resAna) (avaliar depois se isso aqui funciona melhor)
wExp = 2 * np.pi * vetFreqExp  # vetor de frequência em rad/s

# Propriedades do Material
E = 205e+9  # Módulo de elasticidade, em Pa
ro = 7589  # Densidade, em kg/m^3
eta = np.zeros(3)  # Fator de amortecimento estrutural inicialmente definido como nulo

# Geometria
L1 = .152
L2 = .171
L3 = .152  # Alturas de cada andar, em metros
b = 0.15  # Largura da parede da estrutura, em metros
h = 0.98e-3  # Espessura da parede da estrutura, em metros
d = 0.22  # Comprimento de cada andar, em metros
a = 9.74e-3  # Espessura de cada andar, em metros

# Massas
m1 = .523
m2 = .674
m3 = 1.703  # Massas colocadas na estrutura, em kg

# Parte 2 - Obtendo a FRF - Modelo Analítico
mAndar = ro * a * b * d  # Massa de cada andar
m = np.array([m3, m2, 0]) + mAndar  # Vetor que mostra a massa total de cada andar, incluindo as massas colocadas na estrutura
M = m * np.identity(3)

#M = np.diag(m)  # MATRIZ DE MASSA
I = b * h ** 3 / 12  # Momento de inércia da estrutura, em relação ao eixo de aplicação da força
k1 = 15 * E * I / L1 ** 3  # Rigidezes totais das paredes da estrutura, por andar, em N/m
k2 = 15 * E * I / L2 ** 3
k3 = 15 * E * I / L3 ** 3

# Primeiramente, o coeficiente que multiplica os k's era 24. Depois das mudanças, foi para 15.
K = np.array([[k1+k2, -k2, 0], [-k2, k2+k3, -k3], [0, -k3, k3]])  # MATRIZ DE RIGIDEZ

[U, W] = eig(K, M)  # Função do numpy que calcula os autovalores (matriz W) e autovetores (colunas da matriz U)
# W, U = eig(K, M) (testar com essa também)

#print(U)
fnAna = np.sqrt(np.diag(W)[np.diag(W) > 0]) / (2 * np.pi)  # Frequencias naturais do sistema, em Hz

FRF_1_Ana = np.empty(len(wAna), dtype=complex) # Alocação de espaço para as matrizes de FRF para cada Andar
FRF_2_Ana = np.empty(len(wAna), dtype=complex)
FRF_3_Ana = np.empty(len(wAna), dtype=complex)

#FRF_1_Ana = np.array()

# CÁLCULO DAS FRFS (ACELERÂNCIAS) PARA CADA ANDAR A(w) = -w^2*H(w)
for n in range(len(wAna)):
    FRF_1_soma = 0
    FRF_2_soma = 0
    FRF_3_soma = 0
    #print((W[0,0] - wAna[n]**2 + 1j*eta*W[0,0]))
    for j in range(3):
        FRF_1_incr = (-wAna[n]**2 * U[0] * U[j]) / (W[j] - wAna[n]**2 + 1j*eta*W[j])
        FRF_1_soma += FRF_1_incr
       #print(FRF_1_soma)
        FRF_2_incr = (-wAna[n]**2 * U[1] * U[j]) / (W[j] - wAna[n]**2 + 1j*eta*W[j])
        FRF_2_soma += FRF_2_incr
        #print(FRF_2_soma)
        FRF_3_incr = (-wAna[n]**2 * U[2] * U[j]) / (W[j] - wAna[n]**2 + 1j*eta*W[j])
        FRF_3_soma += FRF_3_incr
        #print(FRF_3_soma)

    FRF_1_Ana[n] += FRF_1_soma
    FRF_2_Ana[n] += FRF_2_soma
    FRF_3_Ana[n] += FRF_3_soma

# Parte 3 - Obtendo a FRF - Modelo Experimental
# Supondo que os dados estão no mesmo formato que o código MATLAB original
FRF_1_andar = dados_df["Signal 2 (Real)":"Signal 2 (Imag.)"]  # função para ler os arquivos txt com os dados experimentais
FRF_2_andar = dados_df["Signal 3 (Real)":"Signal 3 (Imag.)"]
FRF_3_andar = dados_df["Signal 4 (Real)":"Signal 4 (Imag.)"]
FRF_1_andar = g * FRF_1_andar[:nPontosExp, :]  # pegar apenas os pontos até a frequência de 40 Hz
FRF_2_andar = g * FRF_2_andar[:nPontosExp, :]
FRF_3_andar = g * FRF_3_andar[:nPontosExp, :]
FRF_1_Exp = FRF_1_andar[:, 1] + FRF_1_andar[:, 2] * 1j  # definir a FRF em parte real (2ª coluna do txt) e imaginária
FRF_2_Exp = FRF_2_andar[:, 1] + FRF_2_andar[:, 2] * 1j # (3ª coluna)
FRF_3_Exp = FRF_3_andar[:, 1] + FRF_3_andar[:, 2] * 1j

# Parte 4 - Amortecimento
# ENCONTRAR OS PICOS
FRF_modulo = np.abs(FRF_2_Exp)  # calcula o módulo da FRF desejada
delta = np.zeros(len(vetFreqExp))  # alocação de espaço para variáveis
deltaLog = np.zeros(len(vetFreqExp))

# O vetor "delta" será igual às variações do módulo FRF das frequências. O vetor "deltaLog" só conterá valores diferentes de zero quando um pico for detectado (entrar no "if" abaixo).
for n in range(1, len(vetFreqExp)):
    delta[n] = FRF_modulo[n] - FRF_modulo[n-1]
    if delta[n] < 0 and delta[n-1] > 0:
        deltaLog[n-1] = np.log10(np.abs(delta[n]))

# Como todos os deltas que entraram no "if" acima são menores que um em módulo seu log será negativo. O vetor "indLog" tem os índices onde o valor de deltaLog não é zero e o vetor "vetLog" tem os valores do deltaLog nesses índices.
indLog = np.nonzero(deltaLog)[0]
vetLog = deltaLog[indLog]

# Agora, encontre o valor máximo de vetLog, cujo índice será o mesmo índice do vetor que contém a frequência máxima: quanto maior o log do delta, maior é o delta. Como os números todos deram negativos, procuramos o delta "menos negativo", mais próximo de zero, e diferente de zero.
min1, ind1 = np.max(vetLog), np.argmax(vetLog)
indiceFreq1 = indLog[ind1]
fnExp1 = vetFreqExp[indiceFreq1]
vetLog[ind1] = -99999

# Isso se repete até que os 3 valores máximos sejam encontrados. Depois que o valor máximo foi encontrado, um valor baixo o suficiente é colocado (-99999 neste caso) para que ele não seja mais o máximo novamente, e para que possamos encontrar o "segundo valor máximo", e assim por diante.
min2, ind2 = np.max(vetLog), np.argmax(vetLog)
indiceFreq2 = indLog[ind2]
fnExp2 = vetFreqExp[indiceFreq2]
vetLog[ind2] = -99999

min3, ind3 = np.max(vetLog), np.argmax(vetLog)
indiceFreq3 = indLog[ind3]
fnExp3 = vetFreqExp[indiceFreq3]

fnExp = np.sort([fnExp1, fnExp2, fnExp3])  # frequências naturais experimentais
indiceFreq = np.sort([indiceFreq1, indiceFreq2, indiceFreq3])

# BANDA DE MEIA POTÊNCIA
# f1
moduloV2 = np.zeros(3)
f1 = np.zeros(3)
for j in range(3):
    moduloV2[j] = FRF_modulo[indiceFreq[j]] / np.sqrt(2)
    trigger = 0
    n = 1
    while trigger < 1:
        if FRF_modulo[n] > moduloV2[j] and n == indiceFreq[j]:
            trigger = 1
            x1 = resFreq * (moduloV2[j] - FRF_modulo[n-1]) / (FRF_modulo[n] - FRF_modulo[n-1])
            f1[j] = vetFreqExp[n-1] + x1
        n += 1

# f2
moduloV2 = np.zeros(3)
f2 = np.zeros(3)
for j in range(3):
    moduloV2[j] = FRF_modulo[indiceFreq[j]] / np.sqrt(2)
    trigger = 0
    n = 1
    while trigger < 1:
        if FRF_modulo[n-1] > moduloV2[j] and n == indiceFreq[j] + 1:
            trigger = 1
            x2 = resFreq * (moduloV2[j] - FRF_modulo[n]) / (FRF_modulo[n-1] - FRF_modulo[n])
            f2[j] = vetFreqExp[n] - x2
        n += 1

# csi
csi = np.array([(f2[j] - f1[j]) / (2 * fnExp[j]) for j in range(3)])

# eta
eta = 2 * csi
erro = (fnAna-fnExp)/fnExp

# PARTE 5 - ATUALIZAR OS VALORES DA FRF ANALÍTICA
for n in range(len(wAna)):
    FRF_1_soma = 0
    FRF_2_soma = 0
    FRF_3_soma = 0
    for j in range(3):
        FRF_1_incr = (-wAna[n]**2 * U[0,j] * U[0,j]) / (W[j,j] - wAna[n]**2 + 1j*eta[j]*W[j,j])
        FRF_1_soma += FRF_1_incr
        FRF_2_incr = (-wAna[n]**2 * U[1,j] * U[0,j]) / (W[j,j] - wAna[n]**2 + 1j*eta[j]*W[j,j])
        FRF_2_soma += FRF_2_incr
        FRF_3_incr = (-wAna[n]**2 * U[2,j] * U[0,j]) / (W[j,j] - wAna[n]**2 + 1j*eta[j]*W[j,j])
        FRF_3_soma += FRF_3_incr
    FRF_1_Ana[n] = FRF_1_soma
    FRF_2_Ana[n] = FRF_2_soma
    FRF_3_Ana[n] = FRF_3_soma

# PARTE 6 - MODELO ANALÍTICO DE AMORTECIMENTO VISCOSO
FRF_1_Visc = np.zeros(len(wAna))  # Alocação de espaço para as matrizes de FRF para cada Andar
FRF_2_Visc = np.zeros(len(wAna))
FRF_3_Visc = np.zeros(len(wAna))
for n in range(len(wAna)):
    FRF_1_soma = 0
    FRF_2_soma = 0
    FRF_3_soma = 0
    for j in range(3):
        FRF_1_incr = (-wAna[n]**2 * U[0,j] * U[0,j]) / (W[j,j] - wAna[n]**2 + 1j*2*csi[j]*np.sqrt(W[j,j])*wAna[n])
        FRF_1_soma += FRF_1_incr
        FRF_2_incr = (-wAna[n]**2 * U[1,j] * U[0,j]) / (W[j,j] - wAna[n]**2 + 1j*2*csi[j]*np.sqrt(W[j,j])*wAna[n])
        FRF_2_soma += FRF_2_incr
        FRF_3_incr = (-wAna[n]**2 * U[2,j] * U[0,j]) / (W[j,j] - wAna[n]**2 + 1j*2*csi[j]*np.sqrt(W[j,j])*wAna[n])
        FRF_3_soma += FRF_3_incr
    FRF_1_Visc[n] = FRF_1_soma
    FRF_2_Visc[n] = FRF_2_soma
    FRF_3_Visc[n] = FRF_3_soma

# Parte 7 - Plotando os gráficos
import matplotlib.pyplot as plt

# Gráficos da Acelerância
plt.figure(1)
plt.semilogy(vetFreqExp, np.abs(FRF_1_Exp), 'r')
plt.hold(True)
plt.semilogy(vetFreqAna, np.abs(FRF_1_Ana), 'g')
plt.semilogy(vetFreqAna, np.abs(FRF_1_Visc), 'b')
plt.hold(False)
plt.legend(['Experimental', 'Analítico (Estrutural)', 'Analítico (Viscoso)'])
plt.xlabel('Frequência [Hz]')
plt.ylabel('Acelerância, em módulo [m/(s^2*N)]')
plt.ylim([1e-4, 1e+2])
plt.grid(True)

# Cálculo da Mobilidade Y(w)=A(w)/(iw)
Mob_1_Exp = FRF_1_Exp / (1j * wExp)
Mob_2_Exp = FRF_2_Exp / (1j * wExp)
Mob_3_Exp = FRF_3_Exp / (1j * wExp)
Mob_1_Ana = FRF_1_Ana / (1j * wAna)
Mob_2_Ana = FRF_2_Ana / (1j * wAna)
Mob_3_Ana = FRF_3_Ana / (1j * wAna)
Mob_1_Visc = FRF_1_Visc / (1j * wAna)
Mob_2_Visc = FRF_2_Visc / (1j * wAna)
Mob_3_Visc = FRF_3_Visc / (1j * wAna)

# Gráficos da Mobilidade
plt.figure(2)
plt.semilogy(vetFreqExp, np.abs(Mob_2_Exp), 'r')
plt.hold(True)
plt.semilogy(vetFreqAna, np.abs(Mob_2_Ana), 'g')
plt.semilogy(vetFreqAna, np.abs(Mob_2_Visc), 'b')
plt.hold(False)
plt.legend(['Experimental', 'Analítico (Estrutural)', 'Analítico (Viscoso)'])
plt.xlabel('Frequência [Hz]')
plt.ylabel('Mobilidade, em módulo [m/(s*N)]')
plt.ylim([1e-5, 1e-1])
plt.grid(True)

# Gráficos das formas modais
U2 = np.zeros((4, 3))
U2[1:, :] = U[:-1, :]
plt.figure(3)
plt.plot(np.arange(4), U2[:, 0], 'r')
plt.hold(True)
plt.plot(np.arange(4), U2[:, 1], 'g')
plt.plot(np.arange(4), U2[:, 2], 'b')
plt.hold(False)
plt.legend(['Primeiro Modo', 'Segundo Modo', 'Terceiro Modo'])
plt.xlabel('Andar')
plt.ylabel('Amplitude de vibração')
plt.grid(True)
plt.xticks([0, 1, 2, 3])

# Diferença percentual entre o analítico estrutural e viscoso
diferenca = np.abs(Mob_3_Ana - Mob_3_Visc) / np.abs(Mob_3_Ana) * 100
plt.figure(4)
plt.xlabel('Frequência [Hz]')
plt.ylabel('Diferença Percentual Entre os Modelos Analíticos')
plt.semilogy(vetFreqAna, diferenca)
plt.grid(True)

plt.show()