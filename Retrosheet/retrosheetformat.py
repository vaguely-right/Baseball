import pandas as pd
import numpy as np
import os

#%%
os.chdir('2010seve')
rsfiles = os.listdir()
infiles = [i for i in rsfiles if '2018' in i]
gmfiles = [i for i in infiles if '.EV' in i]
with open('2018games.txt','w') as f:
    f.write('gameid,date,hometeam,awayteam,gamesite\n')
with open('2018events.txt','w') as f:
    f.write('gameid,inning,battingteam,outs,batter,batterhand,pitcher,pitcherhand,eventtext,eventtype,battereventflag,abflag,hitvalue,outsonplay\n')

for i in gmfiles:
    os.system('bgame -y 2018 -f 0,1,8,7,9 '+i+' >> 2018games.txt')
    os.system('bevent -y 2018 -f 0,2,3,4,10,11,14,15,29,34,35,36,37,40 '+i+' >> 2018events.txt')

os.chdir('..')
os.rename('2010seve/2018games.txt','2018games.txt')
os.rename('2010seve/2018events.txt','2018events.txt')




