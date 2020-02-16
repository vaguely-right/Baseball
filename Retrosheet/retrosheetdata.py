import pandas as pd
import numpy as np


#%%
year = '2018'
gmfile = 'Data\\'+year+'games.txt'
evfile = 'Data\\'+year+'events.txt'
idfile = 'retroID.csv'

gm = pd.read_csv(gmfile)
ev = pd.read_csv(evfile)
pid = pd.read_csv(idfile,index_col=False)
pid['Name'] = pid.First+' '+pid.Last

gm.head()
ev.head()
pid.head()

#Get the gamesite from the game dataframe
ev = ev.merge(gm[['gameid','gamesite']],how='left',on='gameid')

#Get just the end of batter events
ev = ev[ev.battereventflag=='T']

#Create a dictinary for the eventtype codes
eventdict = {0 : 'UNKNOWN',
             1 : 'NOBAT',
             2 : 'BIPOUT',
             3 : 'K',
             4 : 'NOBAT',
             5 : 'NOBAT',
             6 : 'NOBAT',
             7 : 'NOBAT',
             8 : 'NOBAT',
             9 : 'NOBAT',
             10 : 'NOBAT',
             11 : 'NOBAT',
             12 : 'NOBAT',
             13 : 'NOBAT',
             14 : 'UBB',
             15 : 'IBB',
             16 : 'HBP',
             17 : 'OTHER',
             18 : 'OTHER',
             19 : 'BIPOUT',
             20 : 'SNGL',
             21 : 'DBL',
             22 : 'TRPL',
             23 : 'HR',
             24 : 'NOBAT'}

eventdf = pd.DataFrame.from_dict(eventdict,orient='index')
eventdf.columns=['event']

#Assign event abbreviations to every event
ev = ev.merge(eventdf,how='left',left_on='eventtype',right_index=True)

#Specify sacrifice hit and fly events
ev.event[ev.shflag=='T'] = 'SH'
ev.event[ev.sfflag=='T'] = 'SF'

#%%
#Calculate things like PA, AB, WOBA, from the raw stats
df = pd.pivot_table(ev[['batter','event']],index=['batter'],columns=['event'],aggfunc=len,fill_value=0,margins=True)
df = df[:-1]
df = df.merge(pid[['ID','Name']],how='left',left_on='batter',right_on='ID')
df = df[df.All>=100]
df = df.rename(columns={'All':'PA'})

df['AB'] = df.PA-df.UBB-df.HBP-df.IBB-df.SH-df.SF
df = df[['ID','Name','PA','AB','SNGL','DBL','TRPL','HR','UBB','IBB','HBP','K','SF','SH','BIPOUT','OTHER']]
df.sort_values('PA',ascending=False)

df['BA'] = (df.SNGL+df.DBL+df.TRPL+df.HR)/df.AB
df['OBP'] = (df.SNGL+df.DBL+df.TRPL+df.HR+df.UBB+df.IBB+df.HBP)/(df.AB+df.UBB+df.IBB+df.HBP+df.SF)
df['WOBA'] = (0.69*df.UBB + 0.72*df.HBP + 0.89*df.SNGL + 1.27*df.DBL + 1.62*df.TRPL + 2.10*df.HR) / (df.AB + df.UBB + df.SF + df.HBP)
df.sort_values('BA',ascending=False)




#%%
         Code Meaning

          0    Unknown event
          1    No event
          2    Generic out
          3    Strikeout
          4    Stolen base
          5    Defensive indifference
          6    Caught stealing
          7    Pickoff error
          8    Pickoff
          9    Wild pitch
          10   Passed ball
          11   Balk
          12   Other advance
          13   Foul error
          14   Walk
          15   Intentional walk
          16   Hit by pitch
          17   Interference
          18   Error
          19   Fielder's choice
          20   Single
          21   Double
          22   Triple
          23   Home run
          24   Missing play


