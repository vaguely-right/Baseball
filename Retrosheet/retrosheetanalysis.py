import pandas as pd
import numpy as np
from tqdm import tqdm

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
#%%
def get_events(year):
    if type(year)==int:
        year = str(year)
    #Define the files
    gmfile = 'Data\\'+year+'games.txt'
    evfile = 'Data\\'+year+'events.txt'
    idfile = 'retroID.csv'
    #Read the data
    gm = pd.read_csv(gmfile)
    ev = pd.read_csv(evfile)
    pid = pd.read_csv(idfile,index_col=False)
    pid['Name'] = pid.First+' '+pid.Last
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
                 15 : 'BB',
                 16 : 'BB',
                 17 : 'OTHER',
                 18 : 'OTHER',
                 19 : 'BIPOUT',
                 20 : 'SNGL',
                 21 : 'XBH',
                 22 : 'XBH',
                 23 : 'HR',
                 24 : 'NOBAT'}
    eventdf = pd.DataFrame.from_dict(eventdict,orient='index')
    eventdf.columns=['event']
    #Assign event abbreviations to every event
    ev = ev.merge(eventdf,how='left',left_on='eventtype',right_index=True)
    #Specify sacrifice hit and fly events
    ev.event[ev.shflag=='T'] = 'OTHER'
    ev.event[ev.sfflag=='T'] = 'BIPOUT'
    return ev

#%%
#Get the percentages for every year
allevpct = []
for year in tqdm(range(1970,2020)):
    ev = get_events(year)
    df = ev.event.value_counts(normalize=True).to_frame().transpose()
    df.index = [year]
    allevpct.append(df)
evpct = pd.concat(allevpct)


