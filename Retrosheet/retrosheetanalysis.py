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
# Read the constants
fg = pd.read_csv('fgconstants.csv')
fg.set_index('Season',inplace=True)



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
                 15 : 'OTHER',
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
    ev['timesthrough'] = ev.groupby(['gameid','pitcher']).cumcount()//9
#    ev.timesthrough.replace(3,2,inplace=True)
    ev['pitbathand'] = ev.pitcherhand+ev.batterhand
    return ev

def pivot_events(year,split,minpa=0):
    if split not in ['batter','pitcher','gamesite','batterhand','pitcherhand','timesthrough','pitbathand']:
        print('Invalid split index')
        print('Currently supported: batter, pitcher, gamesite, batterhand, pitcherhand, pitbathand, timesthrough')
        return
    ev = get_events(year)
    ptable = pd.pivot_table(ev[[split,'event']],index=[split],columns=['event'],aggfunc=len,fill_value=0,margins=True)
    ptable = ptable[:-1]
    ptable = ptable.rename(columns={'All':'PA'})
    ptable = ptable[['PA','SNGL','XBH','HR','BB','K','BIPOUT','OTHER']]
    ptable.SNGL = ptable.SNGL/ptable.PA
    ptable.XBH = ptable.XBH/ptable.PA
    ptable.HR = ptable.HR/ptable.PA
    ptable.BB = ptable.BB/ptable.PA
    ptable.K = ptable.K/ptable.PA
    ptable.BIPOUT = ptable.BIPOUT/ptable.PA
    ptable.OTHER = ptable.OTHER/ptable.PA
    ptable['AVG'] = (ptable.SNGL+ptable.XBH+ptable.HR)/(ptable.SNGL+ptable.XBH+ptable.HR+ptable.K+ptable.BIPOUT)
    ptable['OBP'] = (ptable.SNGL+ptable.XBH+ptable.HR+ptable.BB)/(ptable.SNGL+ptable.XBH+ptable.HR+ptable.K+ptable.BIPOUT+ptable.BB)
    ptable['WOBA'] = (ptable.SNGL*0.89+ptable.XBH*1.31+ptable.HR*2.10+ptable.BB*0.70)/(1-ptable.OTHER)
    ptable['FIP'] = (ptable.HR*13+ptable.BB*3-ptable.K*2)/(ptable.K+ptable.BIPOUT)*3+3.05
    return ptable

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
# Maybe a more efficient way?
allpct = []
for year in tqdm(range(1970,2020)):
    ev = get_events(year)
    df = ev.groupby(['event']).size()/len(ev)
    df.index = [year]
    allpct.append(df)
pbar = pd.concat(allpct)
#Doesn't work, not really worth it

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
site.sort_values('FIP')

#%%
#To do: lefty/righty batters
bhand = pd.pivot_table(ev[['batterhand','event']],index=['batterhand'],columns=['event'],aggfunc=len,fill_value=0,margins=True)
bhand = bhand[:-1]
bhand = bhand.rename(columns={'All':'PA'})
bhand = bhand[['PA','SNGL','XBH','HR','BB','K','BIPOUT','OTHER']]
bhand.SNGL = bhand.SNGL/bhand.PA
bhand.XBH = bhand.XBH/bhand.PA
bhand.HR = bhand.HR/bhand.PA
bhand.BB = bhand.BB/bhand.PA
bhand.K = bhand.K/bhand.PA
bhand.BIPOUT = bhand.BIPOUT/bhand.PA
bhand.OTHER = bhand.OTHER/bhand.PA
bhand['AVG'] = (bhand.SNGL+bhand.XBH+bhand.HR)/(bhand.SNGL+bhand.XBH+bhand.HR+bhand.K+bhand.BIPOUT)
bhand['OBP'] = (bhand.SNGL+bhand.XBH+bhand.HR+bhand.BB)/(bhand.SNGL+bhand.XBH+bhand.HR+bhand.K+bhand.BIPOUT+bhand.BB)
bhand['WOBA'] = (bhand.SNGL*0.89+bhand.XBH*1.31+bhand.HR*2.10+bhand.BB*0.70)/(1-bhand.OTHER)
bhand['FIP'] = (bhand.HR*13+bhand.BB*3-bhand.K*2)/(bhand.K+bhand.BIPOUT)*3+3.05
bhand

#%%
#To do: lefty/righty pitchers
phand = pd.pivot_table(ev[['batterhand','event']],index=['batterhand'],columns=['event'],aggfunc=len,fill_value=0,margins=True)
phand = phand[:-1]
phand = phand.rename(columns={'All':'PA'})
phand = phand[['PA','SNGL','XBH','HR','BB','K','BIPOUT','OTHER']]
phand.SNGL = phand.SNGL/phand.PA
phand.XBH = phand.XBH/phand.PA
phand.HR = phand.HR/phand.PA
phand.BB = phand.BB/phand.PA
phand.K = phand.K/phand.PA
phand.BIPOUT = phand.BIPOUT/phand.PA
phand.OTHER = phand.OTHER/phand.PA
phand['AVG'] = (phand.SNGL+phand.XBH+phand.HR)/(phand.SNGL+phand.XBH+phand.HR+phand.K+phand.BIPOUT)
phand['OBP'] = (phand.SNGL+phand.XBH+phand.HR+phand.BB)/(phand.SNGL+phand.XBH+phand.HR+phand.K+phand.BIPOUT+phand.BB)
phand['WOBA'] = (phand.SNGL*0.89+phand.XBH*1.31+phand.HR*2.10+phand.BB*0.70)/(1-phand.OTHER)
phand['FIP'] = (phand.HR*13+phand.BB*3-phand.K*2)/(phand.K+phand.BIPOUT)*3+3.05
phand

#%%
#To do: LL/RR/LR/RL matchups
pltn = pd.pivot_table(ev[['pitcherhand','batterhand','event']],index=['pitcherhand','batterhand'],columns=['event'],aggfunc=len,fill_value=0,margins=True)
pltn = pltn[:-1]
pltn = pltn.rename(columns={'All':'PA'})
pltn = pltn[['PA','SNGL','XBH','HR','BB','K','BIPOUT','OTHER']]
pltn.SNGL = pltn.SNGL/pltn.PA
pltn.XBH = pltn.XBH/pltn.PA
pltn.HR = pltn.HR/pltn.PA
pltn.BB = pltn.BB/pltn.PA
pltn.K = pltn.K/pltn.PA
pltn.BIPOUT = pltn.BIPOUT/pltn.PA
pltn.OTHER = pltn.OTHER/pltn.PA
pltn['AVG'] = (pltn.SNGL+pltn.XBH+pltn.HR)/(pltn.SNGL+pltn.XBH+pltn.HR+pltn.K+pltn.BIPOUT)
pltn['OBP'] = (pltn.SNGL+pltn.XBH+pltn.HR+pltn.BB)/(pltn.SNGL+pltn.XBH+pltn.HR+pltn.K+pltn.BIPOUT+pltn.BB)
pltn['WOBA'] = (pltn.SNGL*0.89+pltn.XBH*1.31+pltn.HR*2.10+pltn.BB*0.70)/(1-pltn.OTHER)
pltn['FIP'] = (pltn.HR*13+pltn.BB*3-pltn.K*2)/(pltn.K+pltn.BIPOUT)*3+3.05
pltn[['AVG','OBP','WOBA','FIP']]

#%%
#To do: calculate times through the order for a pitcher
ev = get_events(2019)
gameid = 'ANA201904040'
gm = ev[ev.gameid==gameid]
gm.groupby(['gameid','batter','pitcher']).size()
#This is the cumulative times a pitcher has faced a batter in a game:
gm.groupby(['gameid','batter','pitcher']).cumcount()
#The problem with that is it doesn't account for pinch hitters
#Just do a cumulative count for the total batters faced
gm.groupby(['gameid','pitcher']).cumcount()
#Times through the order is then:
gm.groupby(['gameid','pitcher']).cumcount()//9

#%%
#EDIT: Changed the get_events function to calculate TTO and platoon
tto = pivot_events(2019,'timesthrough')
tto
pltn = pivot_events(2019,'pitbathand')
pltn

#Look at some old timey times through stats

tto = pivot_events(1973,'timesthrough')
tto

#%%
# Get the mean of every season





















