import pickle
import numpy as np

lister = []
for i in range(1,111):
    arr = np.load('../joints/joint_list_' + str(i) + '.npy')
    lis = arr[0]

    dic = {'Hip': np.array([lis[0][0],lis[1][0], lis[2][0]]),
           'RHip': np.array([lis[0][1],lis[1][1], lis[2][1]]),
           'RKnee': np.array([lis[0][2],lis[1][2], lis[2][2]]),
           'RFoot': np.array([lis[0][3],lis[1][3], lis[2][3]]),
           'LHip': np.array([lis[0][4],lis[1][4], lis[2][4]]),
           'LKnee': np.array([lis[0][5],lis[1][5], lis[2][5]]),
           'LFoot': np.array([lis[0][6],lis[1][6], lis[2][6]]),
           'Spine': np.array([lis[0][7],lis[1][7], lis[2][7]]),
           'Thorax': np.array([lis[0][8],lis[1][8], lis[2][8]]),
           'Neck/Nose': np.array([lis[0][9],lis[1][9], lis[2][9]]),
           'Head': np.array([lis[0][10],lis[1][10], lis[2][10]]),
           'LShoulder': np.array([lis[0][11],lis[1][11], lis[2][11]]),
           'LELbow': np.array([lis[0][12],lis[1][12], lis[2][12]]),
           'LWrist': np.array([lis[0][13],lis[1][13], lis[2][13]]),
           'RShoulder': np.array([lis[0][14],lis[1][14], lis[2][14]]),
           'RElbow':np.array([lis[0][15],lis[1][15], lis[2][15]]),
           'RWrist': np.array([lis[0][16],lis[1][16], lis[2][16]])}
    lister.append(dic)

pickle.dump( lister, open( "../joints/joints.pkl", "wb" ))
