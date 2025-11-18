# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:11:48 2023

Not use lag

Spectral análisis

Time 

Orientations  earth reference frame

Heading Euler,

Angular diference

Use the time  and angle

Visualize gyro data to determine a sample just before and just after the turning portion of the recording
I am assuming the average vertical angular velocity between these two points is 0 because there is the same amount of left turning and right turning. This also assumes all segments are aligned to a heading of 0 degrees at these two points.

Rotate gyro data into earth frame (q x g x q^-1, where q is the orientation quaternion, g is angular velocity, and x represents quaternion multiplication)

Retain the vertical (z axis) component between the first and last sample identified in 1.

Subtract the average

Integrate the vertical angular velocity to get heading angle.


@author: burgosp
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation           
import scipy
from . import config    
import math as m        
import scipy.integrate as it 
from . import s02_turning_functions2025 as s2                                     
                   
def euler(quat): 
    nSamples = len(quat)
    eulerAngles = np.zeros((nSamples,3))
    for n in range(nSamples):
        q  = quat[n,:]                                                
        qNorm = np.sqrt(sum(np.real(np.power(q,2))))
        q  = q/qNorm
        rotation = Rotation.from_quat(quat[n,[1,2,3,0]]) # 1,2,3,0This library uses the convention that the rotation angle is in the last element of the quaternion, rather than the first
        eulerAngles[n,:] = rotation.as_euler('zyx') #'zyx'
    return eulerAngles

# https://www.meccanismocomplesso.org/en/hamiltons-quaternions-and-3d-rotation-with-python/
def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z

def q_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)

def qv_mult(q1, v1):
    q2 = (0.0,) + v1
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]


def euler_to_quaternion(phi, theta, psi):
 
    qw = m.cos(phi/2) * m.cos(theta/2) * m.cos(psi/2) + m.sin(phi/2) * m.sin(theta/2) * m.sin(psi/2)
    qx = m.sin(phi/2) * m.cos(theta/2) * m.cos(psi/2) - m.cos(phi/2) * m.sin(theta/2) * m.sin(psi/2)
    qy = m.cos(phi/2) * m.sin(theta/2) * m.cos(psi/2) + m.sin(phi/2) * m.cos(theta/2) * m.sin(psi/2)
    qz = m.cos(phi/2) * m.cos(theta/2) * m.sin(psi/2) - m.sin(phi/2) * m.sin(theta/2) * m.cos(psi/2)
    return [qw, qx, qy, qz]

def quaternion_to_euler(w, x, y, z):
 
    t0 = 2 * (w * x + y * z)
    t1 = 1 - 2 * (x * x + y * y)
    X = m.atan2(t0, t1)
 
    t2 = 2 * (w * y - z * x)
    t2 = 1 if t2 > 1 else t2
    t2 = -1 if t2 < -1 else t2
    Y = m.asin(t2)
     
    t3 = 2 * (w * z + x * y)
    t4 = 1 - 2 * (y * y + z * z)
    Z = m.atan2(t3, t4)
 
    return X, Y, Z    
    
def quaternions_to_euler(quaternions):
    euler_angles = []

    for q in quaternions:
        x, y, z, w = q

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        euler_angles.append([roll, pitch, yaw])
    
    return np.array(euler_angles)   
# filename = './data/20230112-151414507_Turn.h5' #PD2 without head
# filename = "./data/20230330-152433169_Turn.h5" #HC2 without head
# filename = './data/20230425-15152068_Turn.h5' # PD Long 003
# filename = './data/20220111-091805798_Turn.h5' # PD Turn
filename = './data/20230421-124327444_Turn.h5' # HC PB
# filename = './data/20230523-154716794_Turn.h5' # HC Long 203
# filenames = ['./data/20220111-091805798_Turn.h5',
#               './data/20230421-124327444_Turn.h5']

#%% s01
def s01(filename,tth2,plot,offset, turning):
    # config.filename = filename

    h5 = h5py.File(filename, 'r')
    
    
    print("Keys: %s" % h5.keys())
    print("Keys: %s" % h5['Annotations'])
    print("Keys: %s" % h5['Processed'].keys())
    
    print("Keys: %s" % h5['Sensors'].keys())
    
    ishead=1
    
    labels = []
    data = {}
    data2 = {}
    if plot:
        plt.figure()
    
    for i in h5['Sensors'].keys():
        print(i)
        print("Keys: %s" % h5['Sensors'][i]['Configuration'].attrs.get('Label 0'))
        print("Sample Rate : %s" % h5['Sensors'][i]
              ['Configuration'].attrs.get('Sample Rate'))
        se = "%s" % h5['Sensors'][i]['Configuration'].attrs.get(
            'Label 0').decode("utf-8")
        labels.append(se)
        #accelerometer
        a = h5['Sensors'][i]['Accelerometer']
        a = pd.DataFrame(a)
        a.columns = ['ax', 'ay', 'az']

        #gyroscope
        g = h5['Sensors'][i]['Gyroscope']
        g = pd.DataFrame(g)
        g.columns = ['gx', 'gy', 'gz']
        
        # g = 
        # g = np.array(g)
        #time
        t = h5['Sensors'][i]['Time']
        t = pd.DataFrame(t)/1e6    # from microseconds to seconds
        t0 = pd.DataFrame(t).iloc[0, 0]
        t['1'] = t-t0
        ta = np.array(t["1"])
        t.columns = ['time', 'time0']
        
        timetotal = t.time0.iloc[-1]
        # t = np.array(t))
        result = pd.concat([a, g, t], axis=1, join="inner")
        data[se] = result
        
        quat = h5['Processed'][i]['Orientation']
        quat = pd.DataFrame(quat)
        quat = np.array(quat)
        

        tth = 0 # time threshold
        datpost = t.time0>=tth
        # tth2 = 0.2 # time threshold
        datpost2 = t.time0>tth2
        wintime = len(t.time0[~datpost2])#int(tth2*128) #0.5 seconds 128 fps ##add data
        
    
        nqua  =quat[~datpost2,:] #[0:wintime,:] 
        quatn = np.vstack((quat,nqua))
        gy  = np.array(g)
        ngy = gy[~datpost2,:] #[0:wintime,:]
        gyn = np.vstack((gy,ngy))
        tan2 =np.linspace(ta[-1]+(1/128),ta[-1]+(wintime*(1/128))+(1/128),wintime)
        tan = np.hstack((ta,tan2))
        
    #%% Rotate gyro data into earth frame (q x g x q^-1, where q is the orientation quaternion, g is angular velocity, and x represents quaternion multiplication)

       
        datpb = np.zeros([len(quatn),3])
        # for iq in range(len(quat)):
        #     datpb[iq,:] = qv_mult(tuple(quat[iq,:]),tuple(gy[iq,:])) #q x g x q^-1
        for iq in range(len(quatn)):
            datpb[iq,:] = qv_mult(tuple(quatn[iq,:]),tuple(gyn[iq,:])) #q x g x q^-1       
    #%% Retain the vertical (z axis) component between the first and last sample identified in 1.
        dpn1 = np.array(datpost)
        dpn2 = np.array([True]*(wintime+0))
        # addiT= len(dpn2)
        # print(addiT)
        datpostn = np.hstack((dpn1,dpn2))
        datpbzmov = datpb[datpostn,2]
    #%% Subtract the average
        datpbzmov = datpbzmov-np.mean(datpbzmov[0:])#[0:wintime]
        
        # delete the fake portion
        # datpbzmov = datpbzmov[0:-addiT]
    #%% Integrate the vertical angular velocity to get heading angle.
    
    # it.cumtrapz(  y  ,  t, initial=0.0)
        # angles from quat    
        datpbfin = np.rad2deg(it.cumtrapz(datpbzmov, tan))  
        # plt.figure(),
        # plt.subplot(211)

#%% diagonal offset correction option 1 and filtering
        x1 = tan[int(128*(8+5+offset))]
        y1= np.mean(datpbfin[128*(8+offset):128*(18+offset)])
        
        
        y2 = np.mean(datpbfin[-128*(18+offset):-128*(8+offset)])
        x2 = tan[int(-128*((8+5+offset)))]
        
        p1 = x1+y1*1j
        p2 = x2+y2*1j
        ang = np.rad2deg(np.angle(p2-p1))
        if ang <0:
            ang = 180-ang
        
        m = (y1-y2)/(x1-x2) #slope
        print ("angle is " + str(ang))
        b  = (x1*y2 - x2*y1)/(x1-x2) #  y-intercept
        y1b = m*tan[0] + b
        
        y3 = m*tan[-2] + b

        offsetdiag = np.squeeze(np.linspace(y1b, y3, len(datpbfin)))
        # if ang > 10:
        #     datpbfin2 = datpbfin - offsetdiag
            
        datpbfin2 = datpbfin - offsetdiag
                # datpbfin2 = datpbfin2-np.mean(datpbfin2[0:int(128*0.3)])
                # datpbfin2 = datpbfin2-np.mean(datpbfin2)
        # else:
        #     datpbfin2 = datpbfin

        if abs(datpbfin[0]-datpbfin2[0])> 5:
            datpbfin2 = datpbfin2-abs(datpbfin[0]-datpbfin2[0])            

        cutoff, fs, order = 4,128,4

        datpbfin2 = s2.butter_lowpass_filter(datpbfin2, cutoff, fs, order)
        datpbfin2 = datpbfin2-np.mean(datpbfin2[0:4])

        plt.figure()
        plt.plot(tan[0:-1], datpbfin)
        plt.plot(tan[0],y1b,'ro')
        plt.plot(tan[-2],y3,'ro')
        plt.plot(tan[0:-1], datpbfin2)

#%% remove diag offset option 2 and filtering
        # cutoff, fs, order = 0.001,128,4
        # cutoff2=4
        # datpbfin2 = s2.butter_highpass_filter(datpbfin, cutoff, fs, order)
        # #datpbfin2 = s2.butter_lowpass_filter(datpbfin, cutoff2, fs, order)

        # plt.figure()
        # plt.plot(tan[0:-1], datpbfin)
        # plt.plot(tan[0:-1], datpbfin2)    


#%% saving       
        
        # plt.plot(tan[0:-1], datpbfin)
        # plt.title(se)
    
        at1= euler(quat)
        # datpb =  np.unwrap(at1[:,0])        
        # data2[se] = np.unwrap(at1[:,0])
        data2[se] = datpbfin2
        # replace gyroscope by euler
        if ishead:
            print("is head")
        else:
            if se == "Lumbar":
                data["Head"]= result
                data2["Head"]= np.unwrap(at1[:,0])
            
    
        
        
        # plt.figure() #PB 10 27 2023
        # plt.plot(ta,np.unwrap(at1[:,0])) #,discont=1, period=1
        # plt.title(se)
        
        # plt.figure(),plt.plot(ta,at1)
        # plt.title(se)
        # plt.legend(['ax', 'ay', 'az'])  
        
        # plt.legend(['ax', 'ay', 'az'])
        # o.columns = ['ax', 'ay', 'az']
        
    
        # plt.plot(data["Lumbar"].iloc[:, -1], data["Lumbar"].iloc[:, 0:3])
        
        #####EYE LID EYEBROW
        # data["Head"]["gx"] = data["Head"]["gx"] - data["Sternum"]["gx"]
        # data["Head"]["gz"] = data["Head"]["gz"] - data["Sternum"]["gz"]
    
        # data["Sternum"]["gx"] = data["Sternum"]["gx"] - data["Lumbar"]["gx"]
        # data["Sternum"]["gz"] = data["Sternum"]["gz"] - data["Lumbar"]["gz"]
        
        # return data,data2,timetotal,t
        
        # is_moving = np.array([np.nan]*len(t))
        
        # for index in range(len(t)):
        #     is_moving[index] = np.sqrt(a[index].dot(a[index])) > 11  # threshold = 3 m/s/s
        
        # plt.plot(a.y)
        
        # plt.figure() #PB 10 27 2023
        # plt.subplot(211)
    
        # plt.plot(ta,g.gx)
        # plt.plot(ta,g.gy)
        # plt.plot(ta,g.gz)
        # plt.legend(['gx', 'gy', 'gz'])
        
        # plt.plot(ta, data2[se])
        
        
        # plt.plot(is_moving)
    # if plot:
    #     plt.legend(labels)

    return data,data2,timetotal,t, tan[0:-1]

#%% s01b
def s01b(filename,tth2,plot,offset,offset2, turning):
    # config.filename = filename

    h5 = h5py.File(filename, 'r')
    
    
    print("Keys: %s" % h5.keys())
    print("Keys: %s" % h5['Annotations'])
    print("Keys: %s" % h5['Processed'].keys())
    
    print("Keys: %s" % h5['Sensors'].keys())
    
    ishead=1
    
    labels = []
    data = {}
    data2 = {}
    if plot:
        plt.figure()
    
    for i in h5['Sensors'].keys():
        print(i)
        print("Keys: %s" % h5['Sensors'][i]['Configuration'].attrs.get('Label 0'))
        print("Sample Rate : %s" % h5['Sensors'][i]
              ['Configuration'].attrs.get('Sample Rate'))
        se = "%s" % h5['Sensors'][i]['Configuration'].attrs.get(
            'Label 0').decode("utf-8")
        labels.append(se)
        #accelerometer
        a = h5['Sensors'][i]['Accelerometer']
        a = pd.DataFrame(a)
        a.columns = ['ax', 'ay', 'az']

        #gyroscope
        g = h5['Sensors'][i]['Gyroscope']
        g = pd.DataFrame(g)
        g.columns = ['gx', 'gy', 'gz']
        
        # g = 
        # g = np.array(g)
        #time
        t = h5['Sensors'][i]['Time']
        t = pd.DataFrame(t)/1e6    # from microseconds to seconds
        t0 = pd.DataFrame(t).iloc[0, 0]
        t['1'] = t-t0
        ta = np.array(t["1"])
        t.columns = ['time', 'time0']
        
        timetotal = t.time0.iloc[-1]
        # t = np.array(t))
        result = pd.concat([a, g, t], axis=1, join="inner")
        data[se] = result
        
        quat = h5['Processed'][i]['Orientation']
        quat = pd.DataFrame(quat)
        quat = np.array(quat)
        
        if turning:
            selt = (t.time0>10) #& (t.time0<81)
            # plt.figure(),plt.plot(selt)
            a = a[selt]
            g = g[selt]
            t = t[selt]
            ta = ta[selt]
            t.time0 =t.time0-t.time0.iloc[0]
            # plt.figure(),plt.plot(t.time0)
            timetotal = t.time0.iloc[-1]
            result = pd.concat([a, g, t], axis=1, join="inner")
            data[se] = result
            quat = quat[selt]
            
       

         
        
        
        tth = 0 # time threshold
        datpost = t.time0>=tth
        # tth2 = 0.2 # time threshold
        datpost2 = t.time0>tth2
        wintime = len(t.time0[~datpost2])#int(tth2*128) #0.5 seconds 128 fps ##add data
        
    
        nqua  =quat[~datpost2,:] #[0:wintime,:] 
        quatn = np.vstack((quat,nqua))# quat #np.vstack((quat,nqua))
        gy  = np.array(g)
        ngy = gy[~datpost2,:] #[0:wintime,:]
        gyn = np.vstack((gy,ngy))#gy #np.vstack((gy,ngy))
        tan2 =np.linspace(ta[-1]+(1/128),ta[-1]+(wintime*(1/128))+(1/128),wintime)
        tan = np.hstack((ta,tan2))#ta #np.hstack((ta,tan2))
        
    #%% Rotate gyro data into earth frame (q x g x q^-1, where q is the orientation quaternion, g is angular velocity, and x represents quaternion multiplication)

       
        datpb = np.zeros([len(quatn),3])
        # for iq in range(len(quat)):
        #     datpb[iq,:] = qv_mult(tuple(quat[iq,:]),tuple(gy[iq,:])) #q x g x q^-1
        for iq in range(len(quatn)):
            datpb[iq,:] = qv_mult(tuple(quatn[iq,:]),tuple(gyn[iq,:])) #q x g x q^-1       
    #%% Retain the vertical (z axis) component between the first and last sample identified in 1.
        dpn1 = np.array(datpost)
        dpn2 = np.array([True]*(wintime+0))
        # addiT= len(dpn2)
        # print(addiT)
        datpostn = np.hstack((dpn1,dpn2))#dpn1 #np.hstack((dpn1,dpn2))
        datpbzmov = datpb[datpostn,2]
    #%% Subtract the average
        datpbzmov = datpbzmov-np.mean(datpbzmov[0:])#[0:wintime]
        
        # delete the fake portion
        # datpbzmov = datpbzmov[0:-addiT]
    #%% Integrate the vertical angular velocity to get heading angle.
    
    # it.cumtrapz(  y  ,  t, initial=0.0)
        # angles from quat    
        datpbfin = np.rad2deg(it.cumtrapz(datpbzmov, tan))  
        datpbfin = scipy.signal.detrend(datpbfin)

        # plt.figure(),
        # plt.subplot(211)

#%% diagonal offset correction option 1 and filtering
        x1 = tan[int(128*(8+5+offset))]
        y1= np.mean(datpbfin[128*(8+offset):128*(18+offset)])
        
        
        y2 = np.mean(datpbfin[-128*(18+offset2):-128*(8+offset2)])
        x2 = tan[int(-128*((8+5+offset2)))]
        
        p1 = x1+y1*1j
        p2 = x2+y2*1j
        ang = np.rad2deg(np.angle(p2-p1))
        if ang <0:
            ang = 180-ang
        
        m = (y1-y2)/(x1-x2) #slope
        print ("angle is " + str(ang))
        b  = (x1*y2 - x2*y1)/(x1-x2) #  y-intercept
        y1b = m*tan[0] + b
        
        y3 = m*tan[-2] + b

        offsetdiag = np.squeeze(np.linspace(y1b, y3, len(datpbfin)))
        # if ang > 10:
        datpbfin2 = datpbfin - offsetdiag
        
                # datpbfin2 = datpbfin2-np.mean(datpbfin2[0:int(128*0.3)])
                # datpbfin2 = datpbfin2-np.mean(datpbfin2)
        # else:
        #     datpbfin2 = datpbfin

        if abs(datpbfin[0]-datpbfin2[0])> 5:
            datpbfin2 = datpbfin2-abs(datpbfin[0]-datpbfin2[0])            

        cutoff, fs, order = 4,128,4

        datpbfin2 = s2.butter_lowpass_filter(datpbfin2, cutoff, fs, order)
        datpbfin2 = datpbfin2-np.mean(datpbfin2[0:4])
        


        
        
        if plot:
            plt.figure()
            plt.plot(tan[0:-1], datpbfin)
            plt.plot(tan[0],y1b,'ro')
            plt.plot(tan[-2],y3,'ro')
            plt.plot(tan[0:-1], datpbfin2)

#%% remove diag offset option 2 and filtering
        # cutoff, fs, order = 0.001,128,4
        # cutoff2=4
        # datpbfin2 = s2.butter_highpass_filter(datpbfin, cutoff, fs, order)
        # #datpbfin2 = s2.butter_lowpass_filter(datpbfin, cutoff2, fs, order)

        # plt.figure()
        # plt.plot(tan[0:-1], datpbfin)
        # plt.plot(tan[0:-1], datpbfin2)    


#%% saving       
        
        # plt.plot(tan[0:-1], datpbfin)
        # plt.title(se)
    
        at1= euler(quat)
        # datpb =  np.unwrap(at1[:,0])        
        # data2[se] = np.unwrap(at1[:,0])
        data2[se] = datpbfin2
        # replace gyroscope by euler
        if ishead:
            print("is head")
        else:
            if se == "Lumbar":
                data["Head"]= result
                data2["Head"]= np.unwrap(at1[:,0])
            
    
        
        
        # plt.figure() #PB 10 27 2023
        # plt.plot(ta,np.unwrap(at1[:,0])) #,discont=1, period=1
        # plt.title(se)
        
        # plt.figure(),plt.plot(ta,at1)
        # plt.title(se)
        # plt.legend(['ax', 'ay', 'az'])  
        
        # plt.legend(['ax', 'ay', 'az'])
        # o.columns = ['ax', 'ay', 'az']
        
    
        # plt.plot(data["Lumbar"].iloc[:, -1], data["Lumbar"].iloc[:, 0:3])
        
        #####EYE LID EYEBROW
        # data["Head"]["gx"] = data["Head"]["gx"] - data["Sternum"]["gx"]
        # data["Head"]["gz"] = data["Head"]["gz"] - data["Sternum"]["gz"]
    
        # data["Sternum"]["gx"] = data["Sternum"]["gx"] - data["Lumbar"]["gx"]
        # data["Sternum"]["gz"] = data["Sternum"]["gz"] - data["Lumbar"]["gz"]
        
        # return data,data2,timetotal,t
        
        # is_moving = np.array([np.nan]*len(t))
        
        # for index in range(len(t)):
        #     is_moving[index] = np.sqrt(a[index].dot(a[index])) > 11  # threshold = 3 m/s/s
        
        # plt.plot(a.y)
        
        # plt.figure() #PB 10 27 2023
        # plt.subplot(211)
    
        # plt.plot(ta,g.gx)
        # plt.plot(ta,g.gy)
        # plt.plot(ta,g.gz)
        # plt.legend(['gx', 'gy', 'gz'])
        
        # plt.plot(ta, data2[se])
        
        
        # plt.plot(is_moving)
    # if plot:
    #     plt.legend(labels)
    return data,data2,timetotal,t, tan[0:-1]

#%% s01c
def s01c(filename,tth2,plot,offset, turning):
    # config.filename = filename

    h5 = h5py.File(filename, 'r')
    
    
    print("Keys: %s" % h5.keys())
    print("Keys: %s" % h5['Annotations'])
    print("Keys: %s" % h5['Processed'].keys())
    
    print("Keys: %s" % h5['Sensors'].keys())
    
    ishead=1
    
    labels = []
    data = {}
    data2 = {}
    if plot:
        plt.figure()
    
    for i in h5['Sensors'].keys():
        print(i)
        print("Keys: %s" % h5['Sensors'][i]['Configuration'].attrs.get('Label 0'))
        print("Sample Rate : %s" % h5['Sensors'][i]
              ['Configuration'].attrs.get('Sample Rate'))
        se = "%s" % h5['Sensors'][i]['Configuration'].attrs.get(
            'Label 0').decode("utf-8")
        labels.append(se)
        #accelerometer
        a = h5['Sensors'][i]['Accelerometer']
        a = pd.DataFrame(a)
        a.columns = ['ax', 'ay', 'az']

        #gyroscope
        g = h5['Sensors'][i]['Gyroscope']
        g = pd.DataFrame(g)
        g.columns = ['gx', 'gy', 'gz']
        
        # g = 
        # g = np.array(g)
        #time
        t = h5['Sensors'][i]['Time']
        t = pd.DataFrame(t)/1e6    # from microseconds to seconds
        t0 = pd.DataFrame(t).iloc[0, 0]
        t['1'] = t-t0
        ta = np.array(t["1"])
        t.columns = ['time', 'time0']
        
        timetotal = t.time0.iloc[-1]
        # t = np.array(t))
        result = pd.concat([a, g, t], axis=1, join="inner")
        data[se] = result
        
        quat = h5['Processed'][i]['Orientation']
        quat = pd.DataFrame(quat)
        quat = np.array(quat)
        
        
        if turning:
            selt = (t.time0>20) & (t.time0<81)
            plt.figure(),plt.plot(selt)
            a = a[selt]
            g = g[selt]
            t = t[selt]
            t.time0 =t.time0-t.time0.iloc[0]
            plt.figure(),plt.plot(t.time0)
            timetotal = t.time0.iloc[-1]
            result = pd.concat([a, g, t], axis=1, join="inner")
            data[se] = result
            quat = quat[selt]
            

         
        
        
        tth = 0 # time threshold
        datpost = t.time0>=tth
        # tth2 = 0.2 # time threshold
        datpost2 = t.time0>tth2
        wintime = len(t.time0[~datpost2])#int(tth2*128) #0.5 seconds 128 fps ##add data
        
    
        nqua  =quat[~datpost2,:] #[0:wintime,:] 
        quatn = np.vstack((quat,nqua))
        gy  = np.array(g)
        ngy = gy[~datpost2,:] #[0:wintime,:]
        gyn = np.vstack((gy,ngy))
        tan2 =np.linspace(ta[-1]+(1/128),ta[-1]+(wintime*(1/128))+(1/128),wintime)
        tan = np.hstack((ta,tan2))
        
    #%% Rotate gyro data into earth frame (q x g x q^-1, where q is the orientation quaternion, g is angular velocity, and x represents quaternion multiplication)

       
        datpb = np.zeros([len(quatn),3])
        # for iq in range(len(quat)):
        #     datpb[iq,:] = qv_mult(tuple(quat[iq,:]),tuple(gy[iq,:])) #q x g x q^-1
        for iq in range(len(quatn)):
            datpb[iq,:] = qv_mult(tuple(quatn[iq,:]),tuple(gyn[iq,:])) #q x g x q^-1       
    #%% Retain the vertical (z axis) component between the first and last sample identified in 1.
        dpn1 = np.array(datpost)
        dpn2 = np.array([True]*(wintime+0))
        # addiT= len(dpn2)
        # print(addiT)
        datpostn = np.hstack((dpn1,dpn2))
        datpbzmov = datpb[datpostn,2]
    #%% Subtract the average
        datpbzmov = datpbzmov-np.mean(datpbzmov[0:])#[0:wintime]
        
        # delete the fake portion
        # datpbzmov = datpbzmov[0:-addiT]
    #%% Integrate the vertical angular velocity to get heading angle.
    
    # it.cumtrapz(  y  ,  t, initial=0.0)
        # angles from quat    
        datpbfin = np.rad2deg(it.cumtrapz(datpbzmov, tan))  
        # plt.figure(),
        # plt.subplot(211)

#%% diagonal offset correction option 1 and filtering
        x1 = tan[int(128*(8+5+offset))]
        y1= np.mean(datpbfin[128*(8+offset):128*(18+offset)])
        
        
        y2 = np.mean(datpbfin[-128*(18+offset):-128*(8+offset)])
        x2 = tan[int(-128*((8+5+offset)))]
        
        p1 = x1+y1*1j
        p2 = x2+y2*1j
        ang = np.rad2deg(np.angle(p2-p1))
        if ang <0:
            ang = 180-ang
        
        m = (y1-y2)/(x1-x2) #slope
        print ("angle is " + str(ang))
        b  = (x1*y2 - x2*y1)/(x1-x2) #  y-intercept
        y1b = m*tan[0] + b
        
        y3 = m*tan[-2] + b

        offsetdiag = np.squeeze(np.linspace(y1b, y3, len(datpbfin)))
        if ang > 10:
            datpbfin2 = datpbfin - offsetdiag
        
                # datpbfin2 = datpbfin2-np.mean(datpbfin2[0:int(128*0.3)])
                # datpbfin2 = datpbfin2-np.mean(datpbfin2)
        else:
            datpbfin2 = datpbfin

        if abs(datpbfin[0]-datpbfin2[0])> 5:
            datpbfin2 = datpbfin2-abs(datpbfin[0]-datpbfin2[0])            

        cutoff, fs, order = 4,128,4

        datpbfin2 = s2.butter_lowpass_filter(datpbfin2, cutoff, fs, order)
        datpbfin2 = datpbfin2-np.mean(datpbfin2[0:4])

        plt.figure()
        plt.plot(tan[0:-1], datpbfin)
        plt.plot(tan[0],y1b,'ro')
        plt.plot(tan[-2],y3,'ro')
        plt.plot(tan[0:-1], datpbfin2)

#%% remove diag offset option 2 and filtering
        # cutoff, fs, order = 0.001,128,4
        # cutoff2=4
        # datpbfin2 = s2.butter_highpass_filter(datpbfin, cutoff, fs, order)
        # #datpbfin2 = s2.butter_lowpass_filter(datpbfin, cutoff2, fs, order)

        # plt.figure()
        # plt.plot(tan[0:-1], datpbfin)
        # plt.plot(tan[0:-1], datpbfin2)    


#%% saving       
        
        # plt.plot(tan[0:-1], datpbfin)
        # plt.title(se)
    
        at1= euler(quat)
        # datpb =  np.unwrap(at1[:,0])        
        # data2[se] = np.unwrap(at1[:,0])
        data2[se] = datpbfin2
        # replace gyroscope by euler
        if ishead:
            print("is head")
        else:
            if se == "Lumbar":
                data["Head"]= result
                data2["Head"]= np.unwrap(at1[:,0])
            
    
        
        
        # plt.figure() #PB 10 27 2023
        # plt.plot(ta,np.unwrap(at1[:,0])) #,discont=1, period=1
        # plt.title(se)
        
        # plt.figure(),plt.plot(ta,at1)
        # plt.title(se)
        # plt.legend(['ax', 'ay', 'az'])  
        
        # plt.legend(['ax', 'ay', 'az'])
        # o.columns = ['ax', 'ay', 'az']
        
    
        # plt.plot(data["Lumbar"].iloc[:, -1], data["Lumbar"].iloc[:, 0:3])
        
        #####EYE LID EYEBROW
        # data["Head"]["gx"] = data["Head"]["gx"] - data["Sternum"]["gx"]
        # data["Head"]["gz"] = data["Head"]["gz"] - data["Sternum"]["gz"]
    
        # data["Sternum"]["gx"] = data["Sternum"]["gx"] - data["Lumbar"]["gx"]
        # data["Sternum"]["gz"] = data["Sternum"]["gz"] - data["Lumbar"]["gz"]
        
        # return data,data2,timetotal,t
        
        # is_moving = np.array([np.nan]*len(t))
        
        # for index in range(len(t)):
        #     is_moving[index] = np.sqrt(a[index].dot(a[index])) > 11  # threshold = 3 m/s/s
        
        # plt.plot(a.y)
        
        # plt.figure() #PB 10 27 2023
        # plt.subplot(211)
    
        # plt.plot(ta,g.gx)
        # plt.plot(ta,g.gy)
        # plt.plot(ta,g.gz)
        # plt.legend(['gx', 'gy', 'gz'])
        
        # plt.plot(ta, data2[se])
        
        
        # plt.plot(is_moving)
    # if plot:
    #     plt.legend(labels)
    return data,data2,timetotal,t, tan[0:-1]

#%% s01d
def s01d(filename,plot):
    # config.filename = filename

    h5 = h5py.File(filename, 'r')
    
    
    print("Keys: %s" % h5.keys())
    print("Keys: %s" % h5['Annotations'])
    print("Keys: %s" % h5['Processed'].keys())
    
    print("Keys: %s" % h5['Sensors'].keys())
    
    ishead=1
    
    labels = []
    data = {}
    data2 = {}
    if plot:
        plt.figure()
    
    for i in h5['Sensors'].keys():
        print(i)
        print("Keys: %s" % h5['Sensors'][i]['Configuration'].attrs.get('Label 0'))
        print("Sample Rate : %s" % h5['Sensors'][i]
              ['Configuration'].attrs.get('Sample Rate'))
        se = "%s" % h5['Sensors'][i]['Configuration'].attrs.get(
            'Label 0').decode("utf-8")
        labels.append(se)
        #accelerometer
        a = h5['Sensors'][i]['Accelerometer']
        a = pd.DataFrame(a)
        a.columns = ['ax', 'ay', 'az']

        #gyroscope
        g = h5['Sensors'][i]['Gyroscope']
        g = pd.DataFrame(g)
        g.columns = ['gx', 'gy', 'gz']
        
        # g = 
        # g = np.array(g)
        #time
        t = h5['Sensors'][i]['Time']
        t = pd.DataFrame(t)/1e6    # from microseconds to seconds
        t0 = pd.DataFrame(t).iloc[0, 0]
        t['1'] = t-t0
        ta = np.array(t["1"])
        t.columns = ['time', 'time0']
        
        timetotal = t.time0.iloc[-1]
        # t = np.array(t))
        result = pd.concat([a, g, t], axis=1, join="inner")
        data[se] = result
        
        quat = h5['Processed'][i]['Orientation']
        quat = pd.DataFrame(quat)
        quat = np.array(quat)
        
        at1= euler(quat)
        at2= quaternions_to_euler(quat)
        at3 = np.unwrap(at2[:,0])
        at3 = at3 - at3[0]
        plt.figure(), plt.plot(-at3),plt.title(se)
        data2[se]=-at3
        
    return data,data2,timetotal,t
        


def diagonal_correction(datpbfin,pt1,pt2,sr):
    # datpbfin,pt1,pt2,sr   data, time1, time2, sampling_rate
    tan = np.arange(0,len(datpbfin),1)*(1./sr)
    tanpt1 = np.where(tan-pt1==np.min(tan-pt1))[0][0]
    x1 = tan[pt1]
    y1= datpbfin[pt1]
    
    
    tanpt2 = np.where(tan-pt2==np.min(tan-pt2))[0][0]
    x2 = tan[pt2]
    y2= datpbfin[pt2]
    
    p1 = x1+y1*1j
    p2 = x2+y2*1j
    ang = np.rad2deg(np.angle(p2-p1))
    if ang <0:
        ang = 180-ang
    
    m = (y1-y2)/(x1-x2) #slope
    # print ("angle is " + str(ang))
    b  = (x1*y2 - x2*y1)/(x1-x2) #  y-intercept
    y1b = m*tan[0] + b
    
    y3 = m*tan[-1] + b

    offsetdiag = np.squeeze(np.linspace(y1b, y3, len(datpbfin)))
    # if ang > 10:
    #     datpbfin2 = datpbfin - offsetdiag
        
    datpbfin2 = datpbfin - offsetdiag
    return datpbfin2
