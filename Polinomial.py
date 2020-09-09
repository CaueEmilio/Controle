# -*- coding: utf-8 -*-
"""Bibliotecas utilizadas"""

import numpy as np
import matplotlib.pyplot as plt
from math import *
import control as ct
s = ct.TransferFunction.s

"""Classe para encontrar a matriz de sylvester"""

class Sylvester:
    def __init__(self, G):
        """ Recebe duas funções de transferência, uma para o numerador e outra para o denominador """
        self.G = G
        if ct.tfdata(self.G)[1][0][0][::-1].size > ct.tfdata(self.G)[0][0][0][::-1].size:
            self.A = ct.tfdata(self.G)[1][0][0][::-1]
            self.B = ct.tfdata(self.G)[0][0][0][::-1]
        else:
            raise ValueError("O sistema precisa ser causal(Grau Denominador > Grau Numerador).")
            
    def E(self):
        """Mostra a matriz E utilizada para a abrodagem polinomial """
        rows = []
        m = self.A.size
        n = self.B.size
        size = (m-1)*2
        for i in range(size):
            tail = []
            row = []
            if i in range(0, int(size/2)):
                row = list(self.A)
                row.extend((size-m-i)*[0])
                row[:0] = [0]*i
                rows.append(row)
            if i in range(int(size/2), size):
                row = list(self.B)
                row.extend((size-n-i+m-1)*[0])
                row[:0] = [0]*(size-len(row))         
                rows.append(row)
        return np.array(np.transpose(rows))

    def Epid(self,D):
        """ Mostra a matriz E utilizada para a abrodagem polinomial para PID"""
        if len(D)==4:
            B = list(self.B)
            if len(self.B) == 2:
                row1 = [0, B[1],B[0]-B[1]*D[2][0]]
                row2 = [B[1],B[0], -B[1]*D[1][0]]
                row3 = [B[0],0 , -B[1]*D[0][0]]
                return np.array(row1,row2,row3)
            else:
                row1 = [0, 0 ,B[0]]
                row2 = [0 ,B[0], 0]
                row3 = [B[0],0 , 0]
                return np.array([row1,row2,row3])
        else:
            raise ValueError("O polinômio D precisa ter ordem 3")
        

    def PolosOS(self,ts,os):
        """Entre com o tempo de acomodação e a taxa de sobrepassagem em porcentagem para retornar os polos 
        """
        if os != 0:
            zeta = (-np.log(os/100))/(sqrt(pi**2 + log(os/100)**2))
            Wn = 4/(ts*zeta)
            teta = np.arccos(zeta)
            real = -zeta*Wn
            imag = sin(teta)*Wn
            return [complex(real,imag),complex(real,-imag)]
        else: 
            P = -4/ts
            return [complex(P,0)]
  
    def PolosZWOS(self,zw,os):
        """Entre com o Zeta*Wn e a taxa de sobrepassagem em porcentagem para retornar os polos 
        """
        if os != 0:
            zeta = (-np.log(os/100))/(sqrt(pi**2 + log(os/100)**2))
            Wn = zw/zeta
            teta = np.arccos(zeta)
            real = -zw
            imag = sin(teta)*Wn
            return [complex(real,imag),complex(real,-imag)]
        else: 
            P = -4/ts
            return [complex(P,0)]
  
    
    def PolosZeta(self,ts,zeta):
        """Entre com o tempo de acomodação e o zeta para os polos e a taxa de sobrepassagem em porcentagem
        """
        os = exp((-zeta*pi)/sqrt(1-zeta**2))*100
        Wn = 4/(ts*zeta)
        teta = np.arccos(zeta)
        real = -zeta*Wn
        imag = sin(teta)*Wn
        d = (s-complex(real,imag))*(s-complex(real,-imag))
        return [complex(real,imag),complex(real,-imag)]  
    
    def D(self,ts,os):
        """ Encontra a matriz D com base no Tempo de acomodação e da taxa de sobrepassagem"""
        d = self.PolosOS(ts,os)
        i=0
        D = 1
        while len(d) < (self.A.size-1)*2-1:
            d.append(d[0].real*10+i)
            i+=1
        for i in d:
            D = D*(s-i)
        D = np.transpose([ct.tfdata(D)[0][0][0][::-1]]).real
        return D

    def Dzwos(self,zw,os):
        """ Encontra a matriz D com base no Zeta*Wn e da taxa de sobrepassagem"""
        d = self.PolosZWOS(zw,os)
        i=0
        D = 1
        while len(d) < (self.A.size-1)*2-1:
            d.append(d[0].real*10+i)
            i+=1
        for i in d:
            D = D*(s-i)
        D = np.transpose([ct.tfdata(D)[0][0][0][::-1]]).real
        return D


    def C(self,ts,os,D=[]):
        """ Encontra a beta e alfa com base no Tempo de acomodação e da taxa de sobrepassagem"""
        if D == []:
            D = self.D(ts,os)
        else:
            D = D
        invE = np.linalg.inv(self.E())
        M = np.dot(invE,D)
        alpha = np.transpose(M[0:int(M.size/2)][::-1])[0]
        beta = np.transpose(M[int(M.size/2):M.size][::-1])[0]
        return ct.TransferFunction(beta,alpha)

    def Czwos(self,zw,os,D=[]):
        """ Encontra a beta e alfa com base no Zeta*Wn e da taxa de sobrepassagem"""
        if D == []:
            D = self.Dzwos(zw,os)
        else:
            D = D
        invE = np.linalg.inv(self.E())
        M = np.dot(invE,D)
        alpha = np.transpose(M[0:int(M.size/2)][::-1])[0]
        beta = np.transpose(M[int(M.size/2):M.size][::-1])[0]
        return ct.TransferFunction(beta,alpha)

    def Cpid(self,D):
        invE = np.linalg.inv(self.Epid(D))
        A = self.A
        B = [D[2][0]-A[1],D[1][0]-A[0],D[0][0]]
        beta = np.dot(invE,B)[::-1]

        return ct.TransferFunction(beta,[1, 0])
    
"""Teste da classe"""

#G = (s + 2)/(s**2 + s + 0.5)
#G = 3*(s + 4)/(s**3 + 7*s**2 + 11*s -4)
#sylv = Sylvester(G)
#sylv.E()
#sylv.D(8,12)
#sylv.C(8,12)

"""função para encontrar parâmetros do sistema"""

class Parametros:
    """Entre com o tempo de acomodação e a taxa de sobrepassagem em porcentagem ou o coef. de amortecimento (zeta)
    """
    def AcharZeta(ts,os):
      """Entre com o tempo de acomodação e a taxa de sobrepassagem em porcentagem para encontrar o zeta, Wn e o angulo teta em graus
      """
      zeta = (-np.log(os/100))/(sqrt(pi**2 + log(os/100)**2))
      Wn = 4/(ts*zeta)
      teta = np.arccos(zeta)*180/pi
      return print("Zeta = {0:.4f}, Wn = {1:.2f} e teta = {2:.2f}°".format(zeta, Wn, teta))
    
    def AcharOS(ts=5,zeta=0.7):
      """Entre com o tempo de acomodação e o zeta para encontrar a taxa de sobrepassagem em porcentagem, Wn e o angulo teta
      """
      os = exp((-zeta*pi)/sqrt(1-zeta**2))*100
      Wn = 4/(ts*zeta)
      teta = np.arccos(zeta)*180/pi  
      return print("OS% = {0:.2f}, Wn = {1:.2f} e teta = {2:.2f}°".format(os, Wn, teta))

    def PolosOS(ts,os):
        """Entre com o tempo de acomodação e a taxa de sobrepassagem em porcentagem para encontrar o zeta, Wn e o angulo teta em graus
        """
        zeta = (-np.log(os/100))/(sqrt(pi**2 + log(os/100)**2))
        Wn = 4/(ts*zeta)
        teta = np.arccos(zeta)*180/pi
        real = -zeta*Wn
        imag = sin(teta)*Wn
        return print("P1 = {0:.4f} \nP2 = {1:.4f} \nd = {2}°".format(complex(real,imag),complex(real,-imag),(s-complex(real,imag))*(s-complex(real,-imag))))

    def PolosZeta(ts,zeta):
        """Entre com o tempo de acomodação e o zeta para encontrar a taxa de sobrepassagem em porcentagem, Wn e o angulo teta
        """
        os = exp((-zeta*pi)/sqrt(1-zeta**2))*100
        Wn = 4/(ts*zeta)
        teta = np.arccos(zeta)*180/pi  
        real = -zeta*Wn
        imag = sin(teta)*Wn
        return print("P1 = {0:.4f} \nP2 = {1:.4f} \nd = {2}°".format(complex(real,imag),complex(real,-imag),(s-complex(real,imag))*(s-complex(real,-imag))))

#Parametros.AcharZeta(8,12)
#Parametros.PolosOS(8,12)

#Parametros.AcharOS(8,0.5594163911096327)
#Parametros.PolosZeta(8,0.5594163911096327)
