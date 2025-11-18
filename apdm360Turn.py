#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:35:33 2025

@author: pburgos
"""

import numpy as np
import sys

import lib.s01_read_h5_func2024 as s1
import lib.s02_turning_functions2025 as s2
sys.path.insert(0, "lib")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import detrend
from scipy.integrate import trapezoid

import warnings
warnings.filterwarnings('ignore')

import random
                     
random.seed(42)

plot_a= 0  # 2 show clusters k mean detection
plot_b = 0  # 2 show clusters k mean detection

#%% read file

############################# PD
filename = './data/ExamplePD.h5'
pa = 'example_PD001'

############################ HC
# filename = './data/ExampleHC.h5'
# pa = 'example_HC001'

#######################

#%% configure number of clusters
clust1 = 3
clust2 = 9

# set functions inputs
off1= 0
off2 = 0
turnfile = 0
diag = 0
left = 0

data,data2,timetotal,t,tan = s1.s01b(filename,4,0,int(off1),int(off2),turnfile)

df1 = pd.concat([pd.DataFrame(data[x]) for x in data], keys=data.keys(), axis=1)
df1.columns = ['{}'.format(x[0]+"_"+x[1]) for x in df1.columns]
# df1.to_excel(filename+"_imu.xlsx")

# df2 = pd.DataFrame(data2)

df2 = pd.concat([pd.DataFrame(data2[x]) for x in data2], keys=data2.keys(), axis=1)
df2.columns = ['{}'.format(x[0]) for x in df2.columns]
df2["time"]=tan

new_columns=list(df2.columns)
coli = 0
for column_name in new_columns:
    if column_name == 'HEAD':
        new_columns[coli] = 'Head'
    coli +=1
df2.columns = new_columns


if pa in ['example_PD000']:
    diag =1
    
if diag:
    df2.Lumbar = s1.diagonal_correction(df2.Lumbar, 52, 55, 128)
    df2.Sternum  = s1.diagonal_correction(df2.Sternum , 52, 55, 128)
    df2.Head = s1.diagonal_correction(df2.Head, 52, 55, 128)
    

if pa in ['example_PD001','example_HC001']:
    df2 = df2[(df2.time>20)&(df2.time<(df2.time.iloc[-1]-10))]


cutoff, fs, order = 0.001,128,4
cutoff2=5


#%% compute onsets

# df2['time']= np.arange(0,(1/sr)*len(df2),1/sr)


# plot_a= 1  # 2 show clusters


fofig = "./figures/FigOnsets_"


if left:
    onoff2a,m1,clust_indexes1 =s2.onsets2(-df2.Lumbar.values, df2.time.values,clust1,clust2,1,plot_a) # 8 and 7 cluster the best approach
    onoff3a,m2,clust_indexes2 =s2.onsets2(-df2.Lumbar.values, df2.time.values,clust1,clust2,0,plot_a)
    onoff4,onoff5,xval =s2.onsets3(-df2.Lumbar.values, df2.time.values,plot_a)        
else:
    onoff2a,m1,clust_indexes1 =s2.onsets2(df2.Lumbar.values, df2.time.values,clust1,clust2,1,plot_a) # 8 and 7 cluster the best approach
    onoff3a,m2,clust_indexes2 =s2.onsets2(df2.Lumbar.values, df2.time.values,clust1,clust2,0,plot_a)
    onoff4,onoff5,xval =s2.onsets3(df2.Lumbar.values, df2.time.values,plot_a)


if onoff2a[0]-onoff4[0]>200:
    onoff2a = np.hstack([onoff4[0],onoff2a])
else:
    onoff2a[0] = onoff4[0]


    

onoff4 = onoff4[0:len(onoff2a)]
for oo2 in onoff2a:            
    oo2dist = np.abs(onoff4-oo2)
    for ood2 in range(len(oo2dist)):
        if oo2dist[ood2] <100:
            onoff2a[ood2] = onoff4[ood2]
            
            

# plt.figure()
# plt.plot(df2.time.values,df2.Lumbar.values)
# plt.plot(df2.time.values[onoff2a],df2.Lumbar.values[onoff2a],'ro')
# plt.plot(df2.time.values[onoff3a],df2.Lumbar.values[onoff3a],'go')
# plt.title(pa)



onsetsR = pd.DataFrame()
onsetsL = pd.DataFrame()
enblocR = pd.DataFrame()
enblocL= pd.DataFrame()

    
onseg2 = dict()
onseg3 = dict()
for s in ['Head', 'Sternum', 'Lumbar'] :
    print(s)
    # plot_b = 1

    if left:
        onoff2,m1 =s2.onsets2B(-df2[s].values, df2.time.values,clust_indexes1,clust2,1,plot_b) # 8 and 7 cluster the best approach
        onoff3,m2 =s2.onsets2B(-df2[s].values, df2.time.values,clust_indexes2,clust2,0,plot_b) # clu,clu2,updown,plot

    else:
        onoff2,m1 =s2.onsets2B(df2[s].values, df2.time.values,clust_indexes1,clust2,1,plot_b) # 8 and 7 cluster the best approach
        onoff3,m2 =s2.onsets2B(df2[s].values, df2.time.values,clust_indexes2,clust2,0,plot_b) # clu,clu2,updown,plot

    onsetsR[s]= df2.time.values[onoff2]
    onsetsL[s]= df2.time.values[onoff3]
    onseg2[s]=onoff2
    onseg3[s]=onoff3

 
plt.figure()
plt.plot(df2.time.values,df2['Head'].values)
plt.plot(df2.time.values[onseg2['Head']],df2['Head'].values[onseg2['Head']],'ro')
plt.plot(df2.time.values[onseg3['Head']],df2['Head'].values[onseg3['Head']],'go')
plt.plot(df2.time.values,df2['Sternum'].values)
plt.plot(df2.time.values[onseg2['Sternum']],df2['Sternum'].values[onseg2['Sternum']],'ro')
plt.plot(df2.time.values[onseg3['Sternum']],df2['Sternum'].values[onseg3['Sternum']],'go')
plt.plot(df2.time.values,df2['Lumbar'].values)
plt.plot(df2.time.values[onseg2['Lumbar']],df2['Lumbar'].values[onseg2['Lumbar']],'ro')
plt.plot(df2.time.values[onseg3['Lumbar']],df2['Lumbar'].values[onseg3['Lumbar']],'go')
plt.xlabel('time')
plt.ylabel('angle')
plt.title(pa)


    
plt.figure()
n = 2
oRmin = pd.concat([onsetsR.min(axis=1)] * (n+1), axis=1, ignore_index=True)
pc=plt.pcolor(onsetsR.values-oRmin.values,vmin=0,vmax=0.5)
plt.xticks([0.5,1.5,2.5],["Head","Sternum","Lumbar"])
plt.ylabel("Turns")

plt.title('R turns apdm '+pa)

cbar = plt.colorbar(pc)
cbar.set_label('time (s)')
plt.savefig("./figures/onsets/Fig02_"+pa+".png")



plt.figure()
n = 2
oLmin = pd.concat([onsetsL.min(axis=1)] * (n+1), axis=1, ignore_index=True)
pc=plt.pcolor(onsetsL.values-oLmin.values,vmin=0,vmax=0.5)
plt.xticks([0.5,1.5,2.5],["Head","Sternum","Lumbar"])
plt.ylabel("Turns")
# plt.xlabel("Onsets (s)")
plt.title('L turns apdm '+pa)
cbar = plt.colorbar(pc)
cbar.set_label('time (s)')
plt.savefig("./figures/onsets/Fig03_"+pa+".png")

plt.show()

#%% area

areaR = pd.DataFrame()
areaR['index2']=np.arange(0,len(onoff3))
areaR['Sub_ID']=pa
areaL = pd.DataFrame()
areaL['index2']=np.arange(0,len(onoff3))
areaL['Sub_ID']=pa
areaRb = pd.DataFrame()
areaRb['index2']=np.arange(0,len(onoff3))
areaRb['Sub_ID']=pa
areaLb = pd.DataFrame()
areaLb['index2']=np.arange(0,len(onoff3))
areaLb['Sub_ID']=pa


meanoffset=True

if meanoffset:
    df2['Lumbar'] =  (df2['Lumbar'] - df2['Lumbar'].mean())-180
    df2['Sternum'] =  (df2['Sternum'] - df2['Sternum'].mean())-180
    df2['Head'] =  (df2['Head'] - df2['Head'].mean()   )-180

plt.figure()
plt.plot(df2.time,df2.Head)
plt.plot(df2.time,df2.Sternum)
plt.plot(df2.time,df2.Lumbar)


for se1 in ['Head', 'Lumbar'] :
    for se2 in ['Head', 'Sternum', 'Lumbar'] :
        areaR[se1+'_'+se2] = np.nan #arange(0,len(onoff3))
        areaRb[se1+'_'+se2] = np.nan #arange(0,len(onoff3))
        areaL[se1+'_'+se2] = np.nan #arange(0,len(onoff3)) 
        areaLb[se1+'_'+se2] = np.nan #arange(0,len(onoff3))

        idx = 0
        idxb = 0
        if len(onoff2)>len(onoff3):
            for tu1,tu2 in zip(onoff2[0:-1],onoff3):    
                areaR[se1+'_'+se2][idx] = np.mean(np.abs((df2[se1].values[tu1:tu2]) - (df2[se2].values[tu1:tu2])))
                areaRb[se1+'_'+se2][idx] = trapezoid(np.abs((df2[se1].values[tu1:tu2]) - (df2[se2].values[tu1:tu2])),df2.time[tu1:tu2])

                idx += 1
 
            
            for tu1b,tu2b in zip(onoff3,onoff2[1:]):
                areaL[se1+'_'+se2][idxb] = np.mean(np.abs((df2[se1].values[tu1b:tu2b]) - (df2[se2].values[tu1b:tu2b])))
                areaLb[se1+'_'+se2][idxb] = trapezoid(np.abs((df2[se1].values[tu1b:tu2b]) - (df2[se2].values[tu1b:tu2b])),df2.time[tu1b:tu2b])
                idxb += 1
        elif len(onoff2)==len(onoff3):      
            for tu1,tu2 in zip(onoff2,onoff3):    
                areaR[se1+'_'+se2][idx] = np.mean(np.abs((df2[se1].values[tu1:tu2]) - (df2[se2].values[tu1:tu2])))
                areaRb[se1+'_'+se2][idx] = trapezoid(np.abs((df2[se1].values[tu1:tu2]) - (df2[se2].values[tu1:tu2])),df2.time[tu1:tu2])
                idx += 1
 
            
            for tu1b,tu2b in zip(onoff3[0:-1],onoff2[1:]):
                areaL[se1+'_'+se2][idxb] = np.mean(np.abs((df2[se1].values[tu1b:tu2b]) - (df2[se2].values[tu1b:tu2b])))
                areaLb[se1+'_'+se2][idxb] = trapezoid(np.abs((df2[se1].values[tu1b:tu2b]) - (df2[se2].values[tu1b:tu2b])),df2.time[tu1b:tu2b])
                idxb += 1
                
plt.figure()
plt.pcolor(areaRb[["Head_Sternum","Head_Lumbar","Lumbar_Sternum"]],vmin=0,vmax=100)
plt.xticks([0.5,1.5,2.5],["Head_Sternum","Head_Lumbar","Lumbar_Sternum"])
plt.ylabel("Turns")
plt.xlabel("Area")
plt.title('R turns ' + pa)
plt.colorbar()
plt.savefig("./figures/onsets/AREA_R_"+pa+".svg")
                
plt.figure()
plt.pcolor(areaLb[["Head_Sternum","Head_Lumbar","Lumbar_Sternum"]],vmin=0,vmax=100)
plt.xticks([0.5,1.5,2.5],["Head_Stenum","Head_Lumbar","Lumbar_Sternum"])
plt.ylabel("Turns")
plt.xlabel("Area")
plt.title('L turns ' + pa)
plt.colorbar()

plt.savefig("./figures/onsets/AREA_L_"+pa+".svg")



