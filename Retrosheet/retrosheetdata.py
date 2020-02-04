import pandas as pd
import numpy as np

#%%
year = '2018'
gmfile = year+'games.txt'
evfile = year+'events.txt'

gm = pd.read_csv(gmfile)
ev = pd.read_csv(evfile)

gm.head()
ev.head()

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
             14 : 'BB',
             15 : 'IBB',
             16 : 'HBP',
             17 : 'OTHER',
             18 : 'OTHER',
             19 : 'BIPOUT',
             20 : '1B',
             21 : '2B',
             22 : '3B',
             23 : 'HR',
             24 : 'NOBAT'}

eventdf = pd.DataFrame.from_dict(eventdict,orient='index')
eventdf.columns=['event']

ev = ev.merge(eventdf,how='left',left_on='eventtype',right_index=True)

pd.pivot_table(ev[['batter','event']],index=['batter'],columns=['event'],aggfunc=len,fill_value=0,margins=True)

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


