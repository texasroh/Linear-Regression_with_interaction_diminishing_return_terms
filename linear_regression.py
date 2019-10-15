# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from scipy.optimize import least_squares
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2



class LinearRegression:
    
    def __init__(self, dataFrame, target=None, interaction = False, diminishing_return = False, r_columns = None):
        
        columns = dataFrame.columns
                
        if target == None:
            self.target = columns[0]
        else:
            self.target = target
        
        self.columns = columns.drop(target)
        self.x_data = dataFrame[columns].values.T
        self.y_data = dataFrame[target].values
        
        self.n_columns = len(self.columns)
        self.n_c = self.n_columns + 1  # adding intercept
        
        self.interaction = interaction
        if self.interaction:
            self.n_c += (self.n_columns*(self.n_columns-1))//2
        
        self.diminishing_return = diminishing_return
        if self.diminishing_return:
            self.r_position = np.ones(self.n_columns, dtype=int)
            if r_columns == None:
                self.r_columns = self.columns
            else:
                tmp_pos=[]
                tmp_col=[]
                for column in self.columns:
                    if column in r_columns:
                        tmp_pos.append(1)
                        tmp_col.append(column)
                    else:
                        tmp_pos.append(0)
                self.r_position = np.array(tmp_pos, dtype=int)
                self.r_columns = np.array(tmp_col, dtype=str)
        else:
            self.r_position = np.zeros(self.n_columns, dtype = int)
        self.n_r = np.sum(self.r_position)
        self.n_c += self.n_r
        
        self.c_flag = self.make_c_flag()
        
    def make_c_flag(self):
        a = np.ones(self.n_c, dtype=int)
        b = 1 - np.eye(self.n_c, dtype=int)
        flag = np.vstack((a,b))
        return flag
        
        
    def _interaction_terms(self, c, x_data, c_flag):
        inter_terms = 0
        c_idx = self.n_columns
        for i in range(self.n_columns-1):
            for j in range(i+1, self.n_columns):
                inter_terms += x_data[i]*x_data[j]*c[c_idx]*c_flag[c_idx]
                c_idx += 1
        return inter_terms
        
    def _diminishing_return(self, x, r):
        return (1-np.exp(-x*r))/r

        
    def _make_equation(self, c, x_data, c_flag):
        r_idx = -self.n_r - 1
        equ = c[-1]*c_flag[-1]  # adding intercept
        
        
        
        for i in range(self.n_columns):
            if self.r_position[i]:   # diminishing return
                if c_flag[r_idx]:
                    equ += c[i]*c_flag[i]*self._diminishing_return(x_data[i], c[r_idx])
                else:
                    equ += c[i]*c_flag[i]*x_data[i]
                r_idx += 1
            else:   # normal term
                equ += c[i]*c_flag[i]*x_data[i]
            

        if self.interaction:
            equ += self._interaction_terms(c, x_data, c_flag)
        
        return equ
            
        
    def _make_regression_func(self,c, x_data, y_data, c_flag):
        return y_data - self._make_equation(c, x_data, c_flag)

    def _least_square(self, c_flag):
        solv = least_squares(self._make_regression_func, self.c, args = (self.x_data, self.y_data, c_flag))
        return solv
    
    def _llk(self, tmp_pred):
        return -len(self.y_data)/2*(np.log(np.sum(np.power(self.y_data-tmp_pred,2)))+1+np.log(2*np.pi)-np.log(len(self.y_data)))
    
    def _lrt(self, L1, L2):
        return 2*(L1-L2)
    
    def _p_value(self, LRT, df):
        return chi2.sf(LRT, df)
            
    def train(self):        
#        np.random.seed(0)
        self.c = np.random.randn(self.n_c)/1000
#        self.c = np.ones(self.n_c)/1000
        
        solv = []
        prediction=[]
        llk=[]
        lrt = []
        p_value=[]
        for flag in self.c_flag:
            print('flag: ', flag)
            tmp_solv = self._least_square(flag)
            solv.append(tmp_solv.x)
            tmp_pred = self._make_equation(tmp_solv.x, self.x_data, flag)
            prediction.append(tmp_pred)
            tmp_llk = self._llk(tmp_pred)
            llk.append(tmp_llk)
        
        for idx, value in enumerate(llk):
            lrt.append(self._lrt(llk[0], llk[idx]))
            p_value.append(self._p_value(lrt[idx], 1))
        
            
        solv = np.array(solv)
        
        self.prediction = prediction
        self.solv = solv
        self.llk = llk
        self.lrt = lrt
        self.p_value = p_value
    
    
    def summary(self):
        r_idx = -self.n_r - 1
        spec = '\n\nInteraction: ' + str(self.interaction) \
                +'\nDiminishing Return: ' + str(self.diminishing_return) \
                +'\nTarget: '+ self.target \
                +'\nNumber of input variables: ' + str(len(self.columns)) + '\n\n'
        for idx, column in enumerate(self.columns):
            tmp = '\nVar.%d: %s' % (idx+1, column)
            spec += tmp
            if self.r_position[idx]:
                tmp = '\t(Diminishing Return)'
                spec += tmp
        
                
        tmp = '\n\n\n\tInput\t\tCoef.\t\tLLK\t\tLRT\t\tP_value\t\tSig.(95%)\n' \
                +'-'*100
        spec += tmp
        tmp = '\n%19s\t%10f\t%10f\t%10f\t%10f\t%5s' % ('Intercept', self.solv[0][-1], self.llk[-1], self.lrt[-1], self.p_value[-1], str(self.lrt[-1]>3.84))
        spec += tmp
        for i in range(self.n_columns):
            tmp = '\n%19s\t%10f\t%10f\t%10f\t%10f\t%5s' % ('Var.'+str(i+1), self.solv[0][i], self.llk[i+1], self.lrt[i+1], self.p_value[i+1], str(self.lrt[i+1]>3.84))
            spec += tmp
            if self.r_position[i]:
                tmp = '\n%19s\t%10f\t%10f\t%10f\t%10f\t%5s' % ('r-Var.'+str(i+1), self.solv[0][r_idx], self.llk[r_idx], self.lrt[r_idx], self.p_value[r_idx], str(self.lrt[r_idx]>3.84))
                spec += tmp
                r_idx += 1
            
        if self.interaction:
            c_idx = self.n_columns +1
            for i in range(self.n_columns-1):
                for j in range(i+1, self.n_columns):
                    tmp = '\n%19s\t%10f\t%10f\t%10f\t%10f\t%5s' % ('Var.'+str(i+1)+' * Var.'+str(j+1), self.solv[0][c_idx-1], self.llk[c_idx], self.lrt[c_idx], self.p_value[c_idx], str(self.lrt[c_idx]>3.84))
                    spec += tmp
                    c_idx += 1
            
        tmp = '\n\nFull model LLK: ' + str(self.llk[0])
        spec += tmp
                
        self.spec = spec
        print(spec)
        
                
    def plot(self):
        plt_name = ['Full model']
        for column in self.columns:
            plt_name.append(column)
        if self.interaction:
            for i in range(self.n_columns-1):
                for j in range(i+1, self.n_columns):
                    plt_name.append('Test '+self.columns[i] + ' * '+self.columns[j])
        if self.diminishing_return:
            for idx, value in enumerate(self.r_position):
                if value:
                    plt_name.append('Test r-'+self.columns[idx])
                    
        plt_name.append('Test Intercept')
        prediction = self.prediction
        fig, ax = plt.subplots(len(prediction), figsize = (5, 5*len(prediction)))
        for i in range(len(prediction)):
            ax[i].set_title(plt_name[i])
            ax[i].set_xlabel('Prediction')
            ax[i].set_ylabel('y_data')
            ax[i].scatter(prediction[i], self.y_data)
            ax[i].plot([prediction[i].min(), prediction[i].max()],[prediction[i].min(), prediction[i].max()], color='red')
        
        plt.show()
        
        
    
