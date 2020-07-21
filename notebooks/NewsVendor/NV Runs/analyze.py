import pickle, os
import numpy as np

path = os.path.abspath('notebooks\\NewsVendor\\NV Runs')
files = os.listdir(path)
files_30_b = [file for file in files if "NV_Backlog_30" in file]
file_30_b_reward = np.zeros((len(files_30_b),2))
for i,file in enumerate(files_30_b):
    sol = pickle.load(open(path+'\\'+file,'rb'))
    file_30_b_reward[i,0] = np.sum(sol['MIP'].P)
    file_30_b_reward[i,1] = np.sum(sol['Oracle'].P)

print(file_30_b_reward)
np.mean(file_30_b_reward,axis=0)
np.std(file_30_b_reward,axis=0)
