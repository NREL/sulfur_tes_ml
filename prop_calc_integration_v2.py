# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

def convertOutToCsv(fileNameIn, fileNameOut, fakeLineNos):
   
    #Read lines from Fluent output file
    with open(fileNameIn) as f:
        lines = f.readlines()
        
    #Make an array without the first 3 lines and split into 9 columns
    arr = np.empty((len(lines)-fakeLineNos,len(lines[5].split( ))))
   
    #Fill numpy array
    for i in np.arange(len(lines)-fakeLineNos):    #for i in the number of Fluent file lines not including the first 3 lines (0-72042 --> 0-72039)
        str = lines[i+fakeLineNos].split( )        #string with the line starting at line 3, split by spaces
        for j in np.arange(len(str)):              #for j in the number of variables in the strong (0-8)
            arr[i,j] = float(str[j])               #the array(row,column) is equal to the the i row and the j column of the string
   
    #convert array to dataframe, indicate the column names
    df = pd.DataFrame(arr,columns=['Time Step','flow-time','delta-time','iters-per-timestep','t_avg','wall_q_awa','cylinder_q_awa','cyl_temp','initial_temp'])
   
    #write dataframe to csv
    # df.to_csv(fileNameOut, index=False, header=['Time Step','flow-time','delta-time','iters-per-timestep','t_avg','wall_q_awa','cylinder_q_awa'])
    
    #return the dataframe
    #print(df)
    return df



#[None] is a placeholder in a list that represents a null value, in other words, [None]*j gives a list that has the vale 'None' in it j times (the length of the list is j)
#You can then replace the 'None' values in the list with the values assigned by your function instea dof having to append to a list or something else more complicated
def kcalc(t_avg):
    j = len(t_avg)
    k_t_avg=[None] * j
    for i in range(0,j):
        if t_avg[i]<=473.15:
            k_t_avg[i]= 0.481 - 0.0018648*t_avg[i] + (2.4844e-6)*(t_avg[i]**2);
        else:
            k_t_avg[i]= 0.1183 - (2.1626e-6)*t_avg[i] +  (1.6661e-7)*(t_avg[i]**2);
    return k_t_avg
def kcalc_i(Tavg, Tw):
    j = len(Tavg)
    T1 = 473.15
    k_1=[None] * j
    k_2=[None] * j
    k_t_avg=[None] * j
    for i in range(0,j):
        if Tw[i]<=T1 and Tavg[i]<=T1:
            k_t_avg[i]= (0.481*(Tw[i]-Tavg[i]) - (0.0018648/2)*((Tw[i]**2)-(Tavg[i]**2)) + ((2.4844e-6)/3)*((Tw[i]**3)-(Tavg[i]**3))) / (Tw[i]-Tavg[i]);
        elif Tw[i]>T1 and Tavg[i]<=T1:
            k_1[i]= 0.481*(T1-Tavg[i]) - (0.0018648/2)*((T1**2)-(Tavg[i]**2)) + ((2.4844e-6)/3)*((T1**3)-(Tavg[i]**3));
            k_2[i]= 0.1183*(Tw[i]-T1) - ((2.1626e-6)/2)*((Tw[i]**2)-(T1**2)) +  ((1.6661e-7)/3)*((Tw[i]**3)-(T1**3));
            k_t_avg[i] = (k_1[i]+k_2[i])/(Tw[i]-Tavg[i])
        else:
            k_t_avg[i]= (0.1183*(Tw[i]-Tavg[i]) - ((2.1626e-6)/2)*((Tw[i]**2)-(Tavg[i]**2)) +  ((1.6661e-7)/3)*((Tw[i]**3)-(Tavg[i]**3))) / (Tw[i]-Tavg[i]);
    return k_t_avg


def rhocalc(t_avg):
    j = len(t_avg)
    rho_t_avg=[None] * j
    for i in range(0,j):
        if t_avg[i]<=422:
            rho_t_avg[i]= 2137.7 - 0.8487*t_avg[i] ;
        elif t_avg[i]>422 and t_avg[i]<462:
            rho_t_avg[i]= 21125 - 129.29*t_avg[i]+ 0.2885*(t_avg[i]**2) - 0.00021506*(t_avg[i]**3) ;
        else:
            rho_t_avg[i]= 2050.8-0.6240*t_avg[i]          
    return rho_t_avg
def rhocalc_i(Tavg, Tw):
    j = len(Tavg)
    T1 = 422
    T2 = 462
    rho_1 =[None] * j
    rho_2 =[None] * j
    rho_3 =[None] * j
    rho_t_avg=[None] * j
    for i in range(0,j):
        if Tw[i]<=T1 and Tavg[i]<=T1:
            rho_t_avg[i]= (2137.7*(Tw[i]-Tavg[i]) - (0.8487/2)*((Tw[i]**2)-(Tavg[i]**2))) / (Tw[i]-Tavg[i]) ;
        elif T1<Tw[i]<T2 and Tavg[i]<=T1:
            rho_1[i]= 2137.7*(T1-Tavg[i]) - (0.8487/2)*((T1**2)-(Tavg[i]**2));
            rho_2[i]= 21125*(Tw[i]-T1) - (129.29/2)*((Tw[i]**2)-(T1**2))+ (0.2885/3)*((Tw[i]**3)-(T1**3)) - (0.00021506/4)*((Tw[i]**4)-(T1**4));
            rho_t_avg[i] = (rho_1[i]+rho_2[i])/(Tw[i]-Tavg[i])
        elif T1<Tw[i]<T2 and T1<Tavg[i]<T2:
            rho_t_avg[i]= (21125*(Tw[i]-Tavg[i]) - (129.29/2)*((Tw[i]**2)-(Tavg[i]**2))+ (0.2885/3)*((Tw[i]**3)-(Tavg[i]**3)) - (0.00021506/4)*((Tw[i]**4)-(Tavg[i]**4))) / (Tw[i]-Tavg[i]) ;
        elif Tw[i]>=T2 and Tavg[i]<=T1:
            rho_1[i]= 2137.7*(T1-Tavg[i]) - (0.8487/2)*((T1**2)-(Tavg[i]**2));
            rho_2[i]= 21125*(T2-T1) - (129.29/2)*((T2**2)-(T1**2))+ (0.2885/3)*((T2**3)-(T1**3)) - (0.00021506/4)*((T2**4)-(T1**4));
            rho_3[i]= 2050.8*(Tw[i]-T2) - (0.6240/2)*((Tw[i]**2)-(T2**2)); 
            rho_t_avg[i] = (rho_1[i]+rho_2[i]+rho_3[i])/(Tw[i]-Tavg[i])
        elif Tw[i]>=T2 and T1<Tavg[i]<T2:
            rho_2[i]= 21125*(T2-Tavg[i]) - (129.29/2)*((T2**2)-(Tavg[i]**2))+ (0.2885/3)*((T2**3)-(Tavg[i]**3)) - (0.00021506/4)*((T2**4)-(Tavg[i]**4));
            rho_3[i]= 2050.8*(Tw[i]-T2) - (0.6240/2)*((Tw[i]**2)-(T2**2));
            rho_t_avg[i] = (rho_2[i]+rho_3[i])/(Tw[i]-Tavg[i])
        else:
            rho_t_avg[i]= (2050.8*(Tw[i]-Tavg[i]) - (0.6240/2)*((Tw[i]**2)-(Tavg[i]**2))) / (Tw[i]-Tavg[i])         
    return rho_t_avg

def betacalc(t_avg):
    proptable= pd.read_csv("property_data.csv")
    T=proptable['T']
    beta=proptable['beta']
    j = len(t_avg)
    beta_t_avg=[None] * j
    for i in range(0,j):
        index= np.argmin(np.abs(np.array(T)-t_avg[i]))
        beta_t_avg[i]= abs(( ((t_avg[i] - T[index])/ (T[index+1]- T[index]))*(beta[index+1]- beta[index])  ) + beta[index]);
    return beta_t_avg
def betacalc_i(Tavg, Tw):
    j = len(Tavg)
    T1 = 422
    T2 = 462
    beta_1 =[None] * j
    beta_2 =[None] * j
    beta_3 =[None] * j
    beta_t_avg=[None] * j
    for i in range(0,j):
        if Tw[i]<=T1 and Tavg[i]<=T1:
            beta_t_avg[i]= (-np.log(abs(2137.7-(0.8487*Tw[i]))) + np.log(abs(2137.7-(0.8487*Tavg[i]))))/(Tw[i]-Tavg[i]);
        elif T1<Tw[i]<T2 and Tavg[i]<=T1:
            beta_1[i]= -np.log(abs(2137.7-(0.8487*T1))) + np.log(abs(2137.7-(0.8487*Tavg[i])));
            beta_2[i]= -np.log(abs(21125-(129.29*Tw[i])+(0.2885*(Tw[i]**2))-(0.00021506*(Tw[i]**3)))) + np.log(abs(21125-(129.29*T1)+(0.2885*(T1**2))-(0.00021506*(T1**3))));
            beta_t_avg[i] = (beta_1[i]+beta_2[i])/(Tw[i]-Tavg[i])
        elif T1<Tw[i]<T2 and T1<Tavg[i]<T2:
            beta_t_avg[i]= (-np.log(abs(21125-(129.29*Tw[i])+(0.2885*(Tw[i]**2))-(0.00021506*(Tw[i]**3)))) + np.log(abs(21125-(129.29*Tavg[i])+(0.2885*(Tavg[i]**2))-(0.00021506*(Tavg[i]**3)))))/(Tw[i]-Tavg[i]);
        elif Tw[i]>=T2 and Tavg[i]<=T1:
            beta_1[i]= -np.log(abs(2137.7-(0.8487*T1))) + np.log(abs(2137.7-(0.8487*Tavg[i])));
            beta_2[i]= -np.log(abs(21125-(129.29*T2)+(0.2885*(T2**2))-(0.00021506*(T2**3)))) + np.log(abs(21125-(129.29*T1)+(0.2885*(T1**2))-(0.00021506*(T1**3))));
            beta_3[i]= -np.log(abs(2050.8-(0.624*Tw[i]))) + np.log(abs(2050.8-(0.624*T2)));
            beta_t_avg[i] = (beta_1[i]+beta_2[i]+beta_3[i])/(Tw[i]-Tavg[i])
        elif Tw[i]>=T2 and T1<Tavg[i]<T2:
            beta_2[i]= -np.log(abs(21125-(129.29*T2)+(0.2885*(T2**2))-(0.00021506*(T2**3)))) + np.log(abs(21125-(129.29*Tavg[i])+(0.2885*(Tavg[i]**2))-(0.00021506*(Tavg[i]**3))));
            beta_3[i]= -np.log(abs(2050.8-(0.624*Tw[i]))) + np.log(abs(2050.8-(0.624*T2)));
            beta_t_avg[i] = (beta_2[i]+beta_3[i])/(Tw[i]-Tavg[i])
        else:
            beta_t_avg[i]= (-np.log(abs(2050.8-(0.624*Tw[i]))) + np.log(abs(2050.8-(0.624*Tavg[i]))))/(Tw[i]-Tavg[i]);
    return beta_t_avg
    
    
    
def mucalc(t_avg):
    proptable= pd.read_csv("property_data_2.csv")       #read the excel file with the temperature and density properties
    T=proptable['Tmu']                                  #assign the properties from the excel file to variables
    mu=proptable['mu']
    j = len(t_avg)
    mu_t_avg=[None] * j
    #interpolation
    #for every value of t_avg in the Fluent file, subtract t_avg from the array of property T, find the abs value, 
    #then find the index number of the minimum value out of all those values and assign it to the variable 'index'
    for i in range(0,j):
        index= np.argmin(np.abs(np.array(T)-t_avg[i]))  
        mu_t_avg[i]= abs(( ((t_avg[i] - T[index])/ (T[index+1]- T[index]))*(mu[index+1]- mu[index])  ) + mu[index]);
    return mu_t_avg
def mucalc_i(Tavg, Tw):
    proptable= pd.read_csv("property_data_2.csv")       #read the excel file with the temperature and density properties
    T=proptable['Tmu']                                  #assign the properties from the excel file to variables
    mu=proptable['mu']
    j = len(Tavg)
    l = len(T)
    mu_t_avg=[None]*j
    trap=[None]*j
    trap_1=[None]*j
    trap_2=[None]*j
    for i in range(0,j):
        T_array= np.array(T)
        Tavg_index= (np.argmin(T_array < Tavg[i]))-1 
        Tw_index= (np.argmin(T_array < Tw[i]))-1
        mu_Tavg= abs(( ((Tavg[i] - T[Tavg_index])/ (T[Tavg_index+1]- T[Tavg_index]))*(mu[Tavg_index+1]- mu[Tavg_index])  ) + mu[Tavg_index]);
        mu_Tw= abs(( ((Tw[i] - T[Tw_index])/ (T[Tw_index+1]- T[Tw_index]))*(mu[Tw_index+1]- mu[Tw_index])  ) + mu[Tw_index]);
        if Tavg_index == Tw_index:
                mu_t_avg[i] = 0.5*(mu_Tavg+mu_Tw)
        else:
            trap_1[i] = 0.5 * (T[Tavg_index+1]-Tavg[i]) * (mu_Tavg + mu[Tavg_index+1])
            trap_2[i] = 0.5 * (Tw[i]-T[Tw_index]) * (mu[Tw_index] + mu_Tw)
            trap_sum = 0
            for x in range(1,l):
                trap[x] = 0.5 * (T[x]-T[x-1]) * (mu[x]+mu[x-1])
                if (T[x] > T[Tavg_index+1]) and (T[x] <= Tw[i]):
                    trap_sum += trap[x]
            mu_t_avg[i] = (trap_1[i]+trap_sum+trap_2[i])/(Tw[i]-Tavg[i])
    return mu_t_avg


def cpcalc(t_avg):
    j = len(t_avg)                         #j is the length of the t_avg values which is the number of rows in the Fluent file
    cp_t_avg=[None] * j                    #creates a list called cp_t_avg with length j
    for i in range(0,j):                   #for i in the range 0-72039
        if t_avg[i]<=392:                  #if the tavg value in row i fits the criteria, calculate cp this way and replace the 'None' in cp_t_avg[i] with the calculated value
            cp_t_avg[i]= 996.458/1000 ;
        elif t_avg[i]>392 and t_avg[i]<= 431.2 :
            cp_t_avg[i]= ((3.636e-7)*(math.exp(1.925/(t_avg[i]- 440.4)) ) )+ 0.002564*t_avg[i] ;
        elif t_avg[i]>431.2 and t_avg[i]<718:
            cp_t_avg[i]= 1.065+ ( 2.599/(t_avg[i]- 428) )-( 0.3092/((t_avg[i]- 428)**2) ) + (5.911e-9)*((t_avg[i]- 428)**3) ;
        elif t_avg[i]>=718:
            cp_t_avg[i]= 1215.535/1000;
        cp_t_avg[i]=cp_t_avg[i]*1000;   
    return cp_t_avg
def cpcalc_i(Tavg, Tw):
    proptable= pd.read_csv("property_data.csv")       #read the excel file with the temperature and density properties
    T=proptable['T']                                  #assign the properties from the excel file to variables
    cp=proptable['cp']
    j = len(Tavg)
    l = len(T)
    cp_t_avg=[None]*j
    trap=[None]*j
    trap_1=[None]*j
    trap_2=[None]*j
    for i in range(0,j):
        T_array= np.array(T)
        Tavg_index= (np.argmin(T_array < Tavg[i]))-1 
        Tw_index= (np.argmin(T_array < Tw[i]))-1
        cp_Tavg= abs(( ((Tavg[i] - T[Tavg_index])/ (T[Tavg_index+1]- T[Tavg_index]))*(cp[Tavg_index+1]- cp[Tavg_index])  ) + cp[Tavg_index]);
        cp_Tw= abs(( ((Tw[i] - T[Tw_index])/ (T[Tw_index+1]- T[Tw_index]))*(cp[Tw_index+1]- cp[Tw_index])  ) + cp[Tw_index]);
        if Tavg_index == Tw_index:
                cp_t_avg[i] = 0.5*(cp_Tavg+cp_Tw)
        else:
            trap_1[i] = 0.5 * (T[Tavg_index+1]-Tavg[i]) * (cp_Tavg + cp[Tavg_index+1])
            trap_2[i] = 0.5 * (Tw[i]-T[Tw_index]) * (cp[Tw_index] + cp_Tw)
            trap_sum = 0
            for x in range(1,l):
                trap[x] = 0.5 * (T[x]-T[x-1]) * (cp[x]+cp[x-1])
                if (T[x] > T[Tavg_index+1]) and (T[x] <= Tw[i]):
                    trap_sum += trap[x]
            cp_t_avg[i] = (trap_1[i]+trap_sum+trap_2[i])/(Tw[i]-Tavg[i])
    return cp_t_avg


def nucalc(t_avg,q,Tavg,Tw,t_ini):
    D = 0.02;
    k = kcalc(t_avg)
    j = len(t_avg)
    nu_t_avg=[None] * j
    h_avg=[None] * j
    soc=[None] * j
    for i in range(0,j):
        nu_t_avg[i]= (q[i]*D)/(k[i]*(Tw[i]-t_avg[i]));
        h_avg[i]=  (q[i])/(Tw[i]-t_avg[i]);
        if Tw[i] != t_ini:
            soc[i]=(t_ini-Tavg[i])/(t_ini-Tw[i]) 
        else:
            soc[i]=0
    return nu_t_avg,h_avg,soc
def nucalc_i(Tavg,Tw,q,t_ini):
    D = 0.02;
    k = kcalc_i(Tavg,Tw)
    j = len(Tavg)
    nu_t_avg=[None] * j
    h_avg=[None] * j
    soc=[None] * j
    for i in range(0,j):
        nu_t_avg[i]= (q[i]*D)/(k[i]*(Tw[i]-Tavg[i]));
        h_avg[i]=  (q[i])/(Tw[i]-Tavg[i]);
        if Tw[i] != t_ini:
            soc[i]=(t_ini-Tavg[i])/(t_ini-Tw[i]);
        else:
            soc[i]=0
    return nu_t_avg,h_avg,soc


def racalc(t_avg,Tavg,Tw,t_ini):
    g = 9.81;
    D = 0.02;
    beta =betacalc(t_avg)
    mu = mucalc(t_avg)
    k = kcalc(t_avg)
    rho = rhocalc(t_avg)
    cp = cpcalc(t_avg)
    j = len(t_avg)
    ra_t_avg=[None] * j
    for i in range(0,j):
        ra_t_avg[i]= (g*beta[i]*(Tw[i]-Tavg[i])*D*D*D*rho[i]*cp[i]*rho[i])/(mu[i]*k[i]);   
    return ra_t_avg
def racalc_i(Tavg,Tw,t_ini):
    g = 9.81;
    D = 0.02;
    beta =betacalc_i(Tavg,Tw)
    mu = mucalc_i(Tavg,Tw)
    k = kcalc_i(Tavg,Tw)
    rho = rhocalc_i(Tavg,Tw)
    cp = cpcalc_i(Tavg,Tw)
    j = len(Tavg)
    ra_t_avg=[None] * j
    for i in range(0,j):
        ra_t_avg[i]= (g*beta[i]*(Tw[i]-Tavg[i])*D*D*D*rho[i]*cp[i]*rho[i])/(mu[i]*k[i]);   
    return ra_t_avg



# dataset1= convertOutToCsv('ML_550_410.out','ML_550_410.csv',3)  #calling the convert to csv function and assigning the variables, labeling it as dataset1
# col= (dataset1['t_avg'] +550)/2    #variable 'col' equals the dataset1 'tavg' column values averaged with 550 (film temp)
# col2=dataset1['cylinder_q_awa']
# col3= dataset1['t_avg'] 
# col4 = dataset1['cyl_temp']
# t_wall=550;
# t_ini=410;



# k_Tf = kcalc(col)
# rho_Tf = rhocalc(col)
# beta_Tf = betacalc(col)
# mu_Tf = mucalc(col)
# cp_Tf = cpcalc(col)
# nu_Tf,h_Tf,soc_Tf = nucalc(col,col2,col3,col4,t_ini)
# ra_Tf = racalc(col,col4,t_ini)

# k = kcalc_i(col3,col4)
# rho = rhocalc_i(col3,col4)
# beta = betacalc_i(col3,col4)
# mu = mucalc_i(col3,col4)
# cp = cpcalc_i(col3,col4)
# nu,h,soc = nucalc_i(col3,col4,col2,t_ini)
# ra = racalc_i(col3,col4,t_ini)


#assign new columns with new values to dataset1
# dataset1['Tfilm']=col
# dataset1['k(Tf)']=k_Tf
# dataset1['rho(Tf)']=rho_Tf
# dataset1['beta(Tf)']=beta_Tf
# dataset1['mu(Tf)']=mu_Tf
# dataset1['cp(Tf)']=cp_Tf
# dataset1['h(Tf)']=h_Tf
# dataset1['soc(Tf)']=soc_Tf
# dataset1['nu(Tf)']=nu_Tf
# dataset1['ra(Tf)']=ra_Tf

# dataset1['Tw']=col4
# dataset1['Tavg']=col3
# dataset1['k']=k
# dataset1['rho']=rho
# dataset1['beta']=beta
# dataset1['mu']=mu
# dataset1['cp']=cp
# dataset1['h']=h
# dataset1['soc']=soc
# dataset1['nu']=nu
# dataset1['ra']=ra


# dataset1 = dataset1.drop(dataset1.columns[[2,3,4,5,6,7,8]], axis=1)     #drop the original columns 0(Time step), 2(delta-time), .... from dataset1
# dataset1.to_csv('ML_550_410_integration.csv',index=False)                         #export the new dataset to a csv file


import os
from pathlib import Path
# directory = 'C:\Users\mshah2\Documents\Data\ML_Data'
frames = {}

for filename in os.listdir(r"C:\Users\mshah2\OneDrive - NREL\HPC4ei-E16_SES\ML_Training_Data\Fluent_post_Python\Integration edit"):
    if filename.endswith(".out"): 
        f2=os.path.splitext(filename)[0]
        print(f2)
        # extensions = "".join(filename.suffixes)
        # str(filename).removesuffix(extensions)
        # dataset1= convertOutToCsv(filename+".out",filename+".csv",3)
        dataset1= convertOutToCsv(filename,f2+".csv",3)
        
        
        f = open(filename, "r")
        number = float(filename[7:-3])
        # and any other code dealing with file
        
        # number = float(filename[7:-3])
        t_ini=number;
        print(number)
        number = float(filename[3:-8])
        t_wall=number;
        print(number)
        f.close()
        
        
        col= (dataset1['t_avg'] +t_wall)/2    #variable 'col' equals the dataset1 'tavg' column values averaged with 550 (film temp)
        col2=dataset1['cylinder_q_awa']
        col3= dataset1['t_avg'] 
        j = len(col)
        col4 =np.full(j, t_wall)
        # col4 = dataset1['cyl_temp']
        
        # ra_avg=racalc(col,col3,t_wall,t_ini)
        # nu_avg,h_avg,soc=nucalc(col,col2,col3,t_wall,t_ini)
        
        k_Tf = kcalc(col)
        rho_Tf = rhocalc(col)
        beta_Tf = betacalc(col)
        mu_Tf = mucalc(col)
        cp_Tf = cpcalc(col)
        nu_Tf,h_Tf,soc_Tf = nucalc(col,col2,col3,col4,t_ini)
        ra_Tf = racalc(col,col3,col4,t_ini)

        k = kcalc_i(col3,col4)
        rho = rhocalc_i(col3,col4)
        beta = betacalc_i(col3,col4)
        mu = mucalc_i(col3,col4)
        cp = cpcalc_i(col3,col4)
        nu,h,soc = nucalc_i(col3,col4,col2,t_ini)
        ra = racalc_i(col3,col4,t_ini)
        # film temperature -> t_avg -> (T_wall+t_avg)/2  -> chnage ra ->
        # do integration for T_wall and t_avg -> ra -> add column -> rho,cp,k,beta (T)
        #Nu ra -> Di
        
        
        
        dataset1['Tfilm']=col
        dataset1['k(Tf)']=k_Tf
        dataset1['rho(Tf)']=rho_Tf
        dataset1['beta(Tf)']=beta_Tf
        dataset1['mu(Tf)']=mu_Tf
        dataset1['cp(Tf)']=cp_Tf
        dataset1['h(Tf)']=h_Tf
        dataset1['soc(Tf)']=soc_Tf
        dataset1['nu(Tf)']=nu_Tf
        dataset1['ra(Tf)']=ra_Tf

        dataset1['Tw']=col4
        dataset1['Tavg']=col3
        dataset1['k']=k
        dataset1['rho']=rho
        dataset1['beta']=beta
        dataset1['mu']=mu
        dataset1['cp']=cp
        dataset1['h']=h
        dataset1['soc']=soc
        dataset1['nu']=nu
        dataset1['ra']=ra


        dataset1 = dataset1.drop(dataset1.columns[[2,3,4,5,6,7,8]], axis=1)     #drop the original columns 0(Time step), 2(delta-time), .... from dataset1
        # dataset1.to_csv('ML_550_410_integration.csv',index=False)  
        dataset1.to_csv(f2+".csv",index=False)
        # name = os.path.splitext(filename)[0]
        # globals()[name] = dataset1
        # dataset1.name=os.path.splitext(filename)[0]
        frames.update({f2:dataset1})
    else:
        continue
  