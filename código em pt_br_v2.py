import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

# Parte 1 - Configuração do programa
np.set_printoptions(precision=3, suppress=True)
i = 1j  # Unidade complexa
g = 9.81  # Aceleração da gravidade, em m/s^2

# Definição dos parâmetros do modelo analítico
freqMaxAna = 40  # Frequência máxima solicitada no trabalho, em Hz
resAna = 0.005  # Resolução do modelo analítico
vetFreqAna = np.arange(0, freqMaxAna + resAna, resAna)  # Vetor de frequência
wAna = 2 * np.pi * vetFreqAna  # Vetor de frequência em rad/s

# Definição dos parâmetros do modelo experimental
freqMaxExp = 40  # Frequência máxima solicitada no trabalho, em Hz
resFreq = 0.5  # Resolução de frequência do acelerômetro, em Hz
nPontosExp = int(freqMaxExp / resFreq + 1)  # Número de pontos do modelo experimental
vetFreqExp = np.linspace(0, freqMaxExp, nPontosExp)  # Vetor de frequência
wExp = 2 * np.pi * vetFreqExp  # Vetor de frequência em rad/s

# Propriedades do material
E = 205e+9  # Módulo de elasticidade, em Pa
ro = 7589  # Densidade, em kg/m^3
eta = np.zeros(3)  # Fator de amortecimento estrutural inicialmente definido como nulo

# Geometria da estrutura
L1 = .152
L2 = .171
L3 = .152  # Alturas de cada andar, em metros
b = 0.15  # Largura da parede da estrutura, em metros
h = 0.98e-3  # Espessura da parede da estrutura, em metros
d = 0.22  # Comprimento de cada andar, em metros
a = 9.74e-3  # Espessura de cada andar, em metros

# Massas da estrutura
m1 = .523
m2 = .674
m3 = 1.703  # Massas colocadas na estrutura, em kg

# Parte 2 - Obtendo a FRF - Modelo Analítico
mAndar = ro * a * b * d  # Massa de cada andar
m = np.array([m3, m2, 0]) + mAndar  # Vetor que mostra a massa total de cada andar
M = np.diag(m)  # Matriz de massa
I = b * h ** 3 / 12  # Momento de inércia da estrutura
k1 = 15 * E * I / L1 ** 3  # Rigidezes totais das paredes da estrutura, por andar
k2 = 15 * E * I / L2 ** 3
k3 = 15 * E * I / L3 ** 3
K = np.array([[k1+k2, -k2, 0], [-k2, k2+k3, -k3], [0, -k3, k3]])

# Vetor de frequência
wExp = 2 * np.pi * vetFreqExp  # Vetor de frequência em rad/s

# Propriedades do material
E = 205e+9  # Módulo de elasticidade, em Pa
ro = 7589  # Densidade, em kg/m^3
eta = np.zeros(3)  # Fator de amortecimento estrutural inicialmente definido como nulo

# Geometria da estrutura
L1 = .152
L2 = .171
L3 = .152  # Alturas de cada andar, em metros
b = 0.15  # Largura da parede da estrutura, em metros
h = 0.98e-3  # Espessura da parede da estrutura, em metros
d = 0.22  # Comprimento de cada andar, em metros
a = 9.74e-3  # Espessura de cada andar, em metros

# Massas da estrutura
m1 = .523
m2 = .674
m3 = 1.703  # Massas colocadas na estrutura, em kg

# Parte 3 - Obtendo a FRF - Modelo Experimental
mAndar = ro * a * b * d  # Massa de cada andar
m = np.array([m3, m2, 0]) + mAndar  # Vetor que mostra a massa total de cada andar
M = np.diag(m)  # Matriz de massa
I = b * h ** 3 / 12  # Momento de inércia da estrutura
k1 = 15 * E * I / L1 ** 3  # Rigidezes totais das paredes da estrutura, por andar
k2 = 15 * E * I / L2 ** 3
k3 = 15 * E * I / L3 ** 3
K = np.array([[k1+k2, -k2, 0], [-k2, k2+k3, -k3], [0, -k3, k3]])  # Matriz de rigidez

# Cálculo das frequências naturais e formas modais
D, V = eig(K, M)
idx = np.argsort(D)
D = D[idx]
V = V[:, idx]

# Matriz de amortecimento
C = np.zeros((3, 3))
for n in range(3):
    C = C + eta[n] * 2 * np.sqrt(D[n]) * np.outer(V[:, n], V[:, n])

# FRF
H = np.zeros((3, 3, len(wAna)), dtype=complex)
for k in range(len(wAna)):
    H[:, :, k] = np.linalg.inv(-wAna[k]**2 * M + i * wAna[k] * C + K)

# Supondo que os dados estão no mesmo formato que o código MATLAB original
FRF_1_andar = np.loadtxt('21_1_valores.txt')  # função para ler os arquivos txt com os dados experimentais
FRF_2_andar = np.loadtxt('21_2_valores.txt')
FRF_3_andar = np.loadtxt('21_3_valores.txt')
FRF_1_andar = g * FRF_1_andar[:nPontosExp, :]  # pegar apenas os pontosPeço desculpas pela interrupção anterior. Aqui está a continuação do código Python corrigido:

```python
# Pegar apenas os pontos até a frequência de 40 Hz
FRF_2_andar = g * FRF_2_andar[:nPontosExp, :]
FRF_3_andar = g * FRF_3_andar[:nPontosExp, :]
FRF_1_Exp = FRF_1_andar[:, 1] + FRF_1_andar[:, 2] * i  # Definir a FRF em parte real (2ª coluna do txt) e imaginária
FRF_2_Exp = FRF_2_andar[:, 1] + FRF_2_andar[:, 2] * i  # (3ª coluna)
FRF_3_Exp = FRF_3_andar[:, 1] + FRF_3_andar[:, 2] * i

# Encontrar os picos
FRF_modulo = np.abs(FRF_2_Exp)  # Calcula o módulo da FRF desejada
delta = np.zeros(len(vetFreqExp))  # Alocação de espaço para variáveis
deltaLog = np.zeros(len(vetFreqExp))

# O vetor "delta" será igual às variações do módulo FRF das frequências. 
# O vetor "deltaLog" só conterá valores diferentes de zero quando um pico for detectado.
for n in range(1, len(vetFreqExp)):
    delta[n] = FRF_modulo[n] - FRF_modulo[n-1]
    if delta[n] < 0 and delta[n-1] > 0:
        deltaLog[n-1] = np.log10(np.abs(delta[n]))

# O vetor "indLog" tem os índices onde o valor de deltaLog não é zero e o vetor "vetLog" tem os valores do deltaLog nesses índices.
indLog = np.nonzero(deltaLog)[0]
vetLog = deltaLog[indLog]

# Encontrar o valor máximo de vetLog, cujo índice será o mesmo índice do vetor que contém a frequência máxima.
min1, ind1 = np.max(vetLog), np.argmax(vetLog)
indiceFreq1 = indLog[ind1]
fnExp1 = vetFreqExp[indiceFreq1]
vetLog[ind1] = -99999

# Isso se repete até que os 3 valores máximos sejam encontrados. 
# Depois que o valor máximo foi encontrado, um valor baixo o suficiente é colocado (-99999 neste caso) para que ele não seja mais o máximo novamente, e para que possamos encontrar o "segundo valor máximo", e assim por diante.
min2, ind2 = np.max(vetLog), np.argmax(vetLog)
indiceFreq2 = indLog[ind2]
fnExp2 = vetFreqExp[indiceFreq2]
vetLog[ind2] = -99999

min3, ind3 = np.max(vetLog), np.argmax(vetLog)
indiceFreq3 = indLog[ind3]
fnExp3 = vetFreqExp[indiceFreq3]

fnExp = np.sort([fnExp1, fnExp2, fnExp3])  # Frequências naturais experimentais
indiceFreq = np.sort([indiceFreq1, indiceFreq2, indiceFreq3])

# Banda de meia potência
# f1
moduloV2 = np.zeros(3)
f1 = np.zeros(Peço desculpas pela interrupção anterior. Aqui está a continuação do código Python corrigido:

```python
# Banda de meia potência
# f1
moduloV2 = np.zeros(3)
f1 = np.zeros(3)
f2 = np.zeros(3)
for n in range(indiceFreq[0], 0, -1):
    moduloV2[0] = np.abs(FRF_2_Exp[n])
    if moduloV2[0] <= np.abs(FRF_2_Exp[indiceFreq[0]]) / np.sqrt(2):
        f1[0] = vetFreqExp[n]
        break

# f2
for n in range(indiceFreq[0], len(vetFreqExp)):
    moduloV2[0] = np.abs(FRF_2_Exp[n])
    if moduloV2[0] <= np.abs(FRF_2_Exp[indiceFreq[0]]) / np.sqrt(2):
        f2[0] = vetFreqExp[n]
        break

# f3
for n in range(indiceFreq[1], 0, -1):
    moduloV2[1] = np.abs(FRF_2_Exp[n])
    if moduloV2[1] <= np.abs(FRF_2_Exp[indiceFreq[1]]) / np.sqrt(2):
        f1[1] = vetFreqExp[n]
        break

# f4
for n in range(indiceFreq[1], len(vetFreqExp)):
    moduloV2[1] = np.abs(FRF_2_Exp[n])
    if moduloV2[1] <= np.abs(FRF_2_Exp[indiceFreq[1]]) / np.sqrt(2):
        f2[1] = vetFreqExp[n]
        break

# f5
for n in range(indiceFreq[2], 0, -1):
    moduloV2[2] = np.abs(FRF_2_Exp[n])
    if moduloV2[2] <= np.abs(FRF_2_Exp[indiceFreq[2]]) / np.sqrt(2):
        f1[2] = vetFreqExp[n]
        break

# f6
for n in range(indiceFreq[2], len(vetFreqExp)):
    moduloV2[2] = np.abs(FRF_2_Exp[n])
    if moduloV2[2] <= np.abs(FRF_2_Exp[indiceFreq[2]]) / np.sqrt(2):
        f2[2] = vetFreqExp[n]
        break

# Cálculo do fator de amortecimento
etaExp = (f2 - f1) / (2 * fnExp)

# Imprimir os resultados
print('Frequências naturais experimentais (Hz):', fnExp)
print('Fatores de amortecimento experimentais:', etaExp)

# Plotar os gráficos
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(vetFreqExp, np.abs(FRF_2_Exp), label='Experimental')
plt.plot(vetFreqAna, np.abs(H[1, 1, :]), label='Analítico')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Módulo da FRF (m/s²/N)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(vetFreqExp, np.angle(FRF_2_Exp, deg=True), label='Experimental')
plt.plot(vetFreqAna, np.angle(H[1, 1Peço desculpas pela interrupção anterior. Aqui está a continuação do código Python corrigido:

```python
# Plotar os gráficos
plt.subplot(2, 1, 2)
plt.plot(vetFreqExp, np.angle(FRF_2_Exp, deg=True), label='Experimental')
plt.plot(vetFreqAna, np.angle(H[1, 1, :], deg=True), label='Analítico')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Fase da FRF (graus)')
plt.legend()

plt.tight_layout()
plt.show()

