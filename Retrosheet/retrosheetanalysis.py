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
evpct = evpct[['SNGL','XBH','HR','BB','K','BIPOUT','OTHER']]
evpct['AVG'] = (evpct.SNGL+evpct.XBH+evpct.HR)/(evpct.SNGL+evpct.XBH+evpct.HR+evpct.K+evpct.BIPOUT)
evpct['OBP'] = (evpct.SNGL+evpct.XBH+evpct.HR+evpct.BB)/(evpct.SNGL+evpct.XBH+evpct.HR+evpct.K+evpct.BIPOUT+evpct.BB)
evpct['WOBA'] = (evpct.SNGL*0.89+evpct.XBH*1.31+evpct.HR*2.10+evpct.BB*0.70)/(1-evpct.OTHER)

#%%
#Get the data for each batter
bat = pd.pivot_table(ev[['batter','event']],index=['batter'],columns=['event'],aggfunc=len,fill_value=0,margins=True)
bat = bat[:-1]
bat = bat.rename(columns={'All':'PA'})
bat = bat[bat.PA>=100]
idfile = 'retroID.csv'
pid = pd.read_csv(idfile,index_col=False)
pid['Name'] = pid.First+' '+pid.Last
bat = bat.merge(pid[['ID','Name']],how='left',left_on='batter',right_on='ID')
bat = bat[['ID','Name','PA','SNGL','XBH','HR','BB','K','BIPOUT','OTHER']]
bat.SNGL = bat.SNGL/bat.PA
bat.XBH = bat.XBH/bat.PA
bat.HR = bat.HR/bat.PA
bat.BB = bat.BB/bat.PA
bat.K = bat.K/bat.PA
bat.BIPOUT = bat.BIPOUT/bat.PA
bat.OTHER = bat.OTHER/bat.PA
bat['AVG'] = (bat.SNGL+bat.XBH+bat.HR)/(bat.SNGL+bat.XBH+bat.HR+bat.K+bat.BIPOUT)
bat['OBP'] = (bat.SNGL+bat.XBH+bat.HR+bat.BB)/(bat.SNGL+bat.XBH+bat.HR+bat.K+bat.BIPOUT+bat.BB)
bat['WOBA'] = (bat.SNGL*0.89+bat.XBH*1.31+bat.HR*2.10+bat.BB*0.70)/(1-bat.OTHER)
bat.sort_values('WOBA',ascending=False).head(20)
bat[bat.PA>=500].sort_values('WOBA',ascending=False).head(20)

#%%
#Get the data for each pitcher
pit = pd.pivot_table(ev[['pitcher','event']],index=['pitcher'],columns=['event'],aggfunc=len,fill_value=0,margins=True)
pit = pit[:-1]
pit = pit.rename(columns={'All':'PA'})
pit = pit[pit.PA>=100]
idfile = 'retroID.csv'
pid = pd.read_csv(idfile,index_col=False)
pid['Name'] = pid.First+' '+pid.Last
pit = pit.merge(pid[['ID','Name']],how='left',left_on='pitcher',right_on='ID')
pit = pit[['ID','Name','PA','SNGL','XBH','HR','BB','K','BIPOUT','OTHER']]
pit.SNGL = pit.SNGL/pit.PA
pit.XBH = pit.XBH/pit.PA
pit.HR = pit.HR/pit.PA
pit.BB = pit.BB/pit.PA
pit.K = pit.K/pit.PA
pit.BIPOUT = pit.BIPOUT/pit.PA
pit.OTHER = pit.OTHER/pit.PA
pit['FIP'] = (pit.HR*13+pit.BB*3-pit.K*2)/(pit.K+pit.BIPOUT)*3+3.05
pit.sort_values('FIP').head(20)
pit[pit.PA>=500].sort_values('FIP').head(20)

#%%
#Get the data for each stadium
site = pd.pivot_table(ev[['gamesite','event']],index=['gamesite'],columns=['event'],aggfunc=len,fill_value=0,margins=True)
site = site[:-1]
site = site.rename(columns={'All':'PA'})
site = site[site.PA>=100]
idfile = 'parkcode.csv'
pid = pd.read_csv(idfile,index_col=False)
site = site.merge(pid[['PARKID','NAME']],how='left',left_on='gamesite',right_on='PARKID')
site = site[['PARKID','NAME','PA','SNGL','XBH','HR','BB','K','BIPOUT','OTHER']]
site.SNGL = site.SNGL/site.PA
site.XBH = site.XBH/site.PA
site.HR = site.HR/site.PA
site.BB = site.BB/site.PA
site.K = site.K/site.PA
site.BIPOUT = site.BIPOUT/site.PA
site.OTHER = site.OTHER/site.PA
site['AVG'] = (site.SNGL+site.XBH+site.HR)/(site.SNGL+site.XBH+site.HR+site.K+site.BIPOUT)
site['OBP'] = (site.SNGL+site.XBH+site.HR+site.BB)/(site.SNGL+site.XBH+site.HR+site.K+site.BIPOUT+site.BB)
site['WOBA'] = (site.SNGL*0.89+site.XBH*1.31+site.HR*2.10+site.BB*0.70)/(1-site.OTHER)
site['FIP'] = (site.HR*13+site.BB*3-site.K*2)/(site.K+site.BIPOUT)*3+3.05

#%%
#To do: lefty/righty batters

#%%
#To do: lefty/righty pitchers

#%%
#To do: LL/RR/LR/RL matchups

#%%
#To do: calculate times through the order for a pitcher

#%%
#To do: summarize by TTO


















