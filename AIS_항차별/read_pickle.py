import pickle

file = '강원해양호_항적'

with open('./Output/voyage_{}.pickle'.format(file),'rb') as fr:
    data = pickle.load(fr)
    
k=1