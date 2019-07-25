__author__ = 'abhishekyanamandra'
__ver__ = 0.1

import pylab
import os
import numpy as np
import pylab
import imageio
import numpy as np
from scipy.misc import imresize
import face_recognition as fr
from tqdm import tqdm_notebook as tqdm
import pickle as p


def path2faces(path, persec=5, size=(100,200), facerec=True):
    vid = imageio.get_reader(path, 'ffmpeg', mode='I')
    frms = np.linspace(0, 300, 50).astype(int)
    IMAGES = []
    for frm in frms:
        try:
            img = vid.get_data(frm)
            img = imresize(img,size=size)
            if facerec:
                size=(50,50)
                fcxy = fr.face_locations(img)[0]
                img = img[fcxy[0]:fcxy[2], fcxy[3]:fcxy[1],:]
            IMAGES.append(img)
        except Exception as e:
            print(e,len(IMAGES))
            return np.asarray(IMAGES)

if __name__=="__main__":
	pass
	return None

c=0
L=[]
for i in sorted(os.listdir('./extracted/')):
    c+=len(sorted(os.listdir('./extracted/'+i)))
    L.extend(['./extracted/'+i+'/'+k for k in sorted(os.listdir('./extracted/'+i))])
print(f"Count is {c} and len of L is {len(L)}")

DATA = {}
for i in tqdm(L):
    DATA[i]=path2faces(i, facerec=False)
p.dump(DATA,open('Full.d','wb'))
FACES = {}
cc=0
for k in tqdm(sorted(list(DATA.keys()))):
    TT=[]
    t=DATA[k]
    for imgg in list(t):
        cc+=1
        try:
            fcxy = fr.face_locations(imgg)
            fcxy = fcxy[0]
            imgg = imgg[fcxy[0]:fcxy[2], fcxy[3]:fcxy[1],:]
            imgg = imresize(imgg,size=(50,50))
            TT.append(imgg)
        except Exception as e:
            print(e,f'at {cc}')
    FACES[k.split('/')[-1].split('.')[0]]=TT
p.dump(FACES,open('Faces.d','wb'))
F = p.load(open('./DATA/Faces.d','rb'))
Paths=[]
for i,v in enumerate(sorted(list(F.keys()))):
    if v[:2]=='01': 
        print(i,v,np.array(F[v]).shape)
        Paths.append(v)

def LoadAudio(paths,F,sr=16000):
    DATA = {}
    for path in tqdm(paths):
        pF = path
        key = '03'+path[2:]
        ac = path[-2:]
        path = Audpath+'Actor_'+ac+'/03'+path[2:]+'.wav'
        t=F[pF]
#         print(key,pF)
        DATA[pF] = [librosa.load(path,sr=sr)[0],np.array(t)]
    return DATA

data = LoadAudio(Paths,F)
p.dump(data,open('./DATA/DATA.d','wb'))

print("Done Preprocessing")