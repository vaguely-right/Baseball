import pandas as pd
import numpy as np
import os
from tqdm import tqdm

#%% Go to the data folder and compile everything
year = '2017'
indir = 'C:\\Data\\retrosheet'
outdir = 'C:\\Users\\s_lys\\Documents\\Python Scripts\\Baseball\\Retrosheet'
os.chdir(indir)
rsfiles = os.listdir()
infiles = [i for i in rsfiles if year in i]
gmfiles = [i for i in infiles if '.EV' in i]
with open(year+'games.txt','w') as f:
    f.write('gameid,date,hometeam,awayteam,gamesite\n')
with open(year+'events.txt','w') as f:
    f.write('gameid,inning,battingteam,outs,batter,batterhand,pitcher,pitcherhand,eventtext,eventtype,battereventflag,abflag,hitvalue,outsonplay\n')

for i in tqdm(gmfiles):
    os.system('bgame -y '+year+' -f 0,1,8,7,9 '+i+' >> '+year+'games.txt')
    os.system('bevent -y '+year+' -f 0,2,3,4,10,11,14,15,29,34,35,36,37,40 '+i+' >> '+year+'events.txt')

os.chdir(outdir)
os.rename(indir+'\\'+year+'games.txt',year+'games.txt')
os.rename(indir+'\\'+year+'events.txt',year+'events.txt')


#NOTE: There is something wrong with the file 1993BAL.EVA
#This file seems to be corrupt in both the 1990seve.zip and 1993eve.zip downloads
#%%
gmfile = '2018games.txt'
evfile = '2018events.txt'

gm = pd.read_csv(gmfile)
ev = pd.read_csv(evfile)

gm.head()
ev.head()

#Looks good!




