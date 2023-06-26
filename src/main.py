import pandas as pd
import numpy as np
from scipy.linalg import eig
import matplotlib
from matplotlib import pyplot as plt

matplotlib.rcParams['figure.dpi']=600 # config


#############
# [14:17, 12/06/2023] +55 48 9179-2999: Dimensões do andar: 226x151,80x9,65mm
# [14:18, 12/06/2023] +55 48 9179-2999: Perfil da viga:
# 49,66x3,2mm
# [14:23, 12/06/2023] +55 48 9179-2999: Distância entre vigas:
# 55mm
# [14:24, 12/06/2023] +55 48 9179-2999: Altura entre andares 0-1/1-2/2-3:
# 120mm / 90,03 mm / 90,67 mm
##############


xf = 1 # andar do shaker
N = 3 # graus de liberdade

# teste
m_eq = 15000
k_eq = 7.5e+7


M = m_eq * np.identity(N) # Matriz massa 
K = k_eq * np.asarray([[2,-1,0], [-1,2,-1],[0,-1,1]]) # Matriz rigidez

# Definição de um amortecimento infimo para evitar |H(omega)|-> infinito
zeta = 1e-4

# Extração dos auto-valores e auto-vetores
[W,V]=eig(K,M)
print(W)
# Normalização dos auto-vetores pela massa
V= V / m_eq**.5

# Calculo de W = sqrt(W^2)
W=np.sqrt(W) 

# teste
#print(W, V)


#Parte III - Cálculo das funções resposta em frequência (correspondem à primeira coluna da matriz de transferência, pois todas as entradas de força, exceto f1 são nulas, não havendo assim utilidade)

# Definição do vetor do domínio para omega, de um valor tendendo 
# a zero até 20% a mais da maior frequência natural
w_domain=2*np.pi*np.linspace(.01,1.2*max(np.real(W)),num=int(1e+3))

# Inicialização de uma matriz em que cada coluna corresponde a 
# um grau de liberdade e cada linha corresponde ao valor da 
# função de transferência em relação a uma força harmônica 
# aplicada ao primeiro elemento avaliada para cada valor de w_domain
Hs = (0+0j)*np.ones([len(w_domain),N])

# Uma matriz análoga, mas contendo os ângulos de fase para cada valor
Hs_phase = np.empty([len(w_domain),N])

#Loop para calcular os valores da função de transferência
for x in range(N):
    for i in range(len(w_domain)):
        S=0
        for j in range(N):
            S=S+ V[x,j]*V[xf-1,j]/(W[j]**2 - w_domain[i]**2 + 2j * zeta * W[j]*w_domain[i])

        Hs[i,x] = S
        Hs_phase[i,x]=np.arctan2(np.imag(S),np.real(S))

print(Hs, Hs_phase)
#Parte IV - Plotagem dos dados

#Alterar a resolução da imagem e algumas definições auxiliares


inf, sup = np.min(np.abs(Hs)) , np.max(np.abs(Hs))
margem = .1*abs(sup - inf)

inf2, sup2 = np.min(Hs_phase) , np.max(Hs_phase)
inf2,sup2,margem2=np.rad2deg([inf2,sup2, .1* abs( sup2- inf2)])

#Plotagem dos valores - loop para cada G.L. do sistema
for i in range(N):

    plt.figure(i)
    plt.subplot(2,1,1)
    plt.loglog(w_domain,np.abs(Hs[:,i]), color='blue')
    plt.xlabel('\N{GREEK SMALL LETTER OMEGA} [ rad / s ]')
    plt.ylabel('| H(\N{GREEK SMALL LETTER OMEGA}) |')
    plt.grid(bool)
    plt.ylim(inf, sup + margem)
    plt.xlim(min(w_domain),max(w_domain))
    print("Aqui")

    plt.subplot(2,1,2)
    plt.semilogx(w_domain,np.rad2deg(Hs_phase[:,i]),color='blue')
    plt.xlabel('\N{GREEK SMALL LETTER OMEGA} [ rad / s ]')
    plt.ylabel('Ângulo de fase \N{GREEK SMALL LETTER PHI} [ graus ]')
    plt.xlim(min(w_domain),max(w_domain))
    plt.ylim(inf2-margem2 , sup2 + margem2)
    plt.grid(bool)

    plt.savefig('FT'+str(i+1)+'.png')
