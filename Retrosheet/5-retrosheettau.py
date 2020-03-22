import pandas as pd
import numpy as np
from tqdm import tqdm
import numpy.linalg as la
import seaborn as sns

pd.set_option('display.width',150)
pd.set_option('display.max_columns',16)

#%% Some functions
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
    ev.timesthrough[ev.timesthrough>2] = 2
    ev['pitbathand'] = ev.pitcherhand+ev.batterhand
    return ev

def pivot_events(year,split,minpa=0):
    if split not in ['batter','pitcher','gamesite','batterhand','pitcherhand','timesthrough','pitbathand']:
        print('Invalid split index')
        print('Currently supported: batter, pitcher, gamesite, batterhand, pitcherhand, pitbathand, timesthrough')
        return
    ev = get_events(year)
# New in this version: drop OTHER events
    ev = ev[ev.event!='OTHER']
    ptable = pd.pivot_table(ev[[split,'event']],index=[split],columns=['event'],aggfunc=len,fill_value=0,margins=True)
    ptable = ptable[:-1]
    ptable = ptable.rename(columns={'All':'PA'})
#    ptable = ptable[['PA','SNGL','XBH','HR','BB','K','BIPOUT','OTHER']]    
    ptable = ptable[['PA','SNGL','XBH','HR','BB','K','BIPOUT']]
    ptable.SNGL = ptable.SNGL/ptable.PA
    ptable.XBH = ptable.XBH/ptable.PA
    ptable.HR = ptable.HR/ptable.PA
    ptable.BB = ptable.BB/ptable.PA
    ptable.K = ptable.K/ptable.PA
    ptable.BIPOUT = ptable.BIPOUT/ptable.PA
    #    ptable.OTHER = ptable.OTHER/ptable.PA
#    ptable['AVG'] = (ptable.SNGL+ptable.XBH+ptable.HR)/(ptable.SNGL+ptable.XBH+ptable.HR+ptable.K+ptable.BIPOUT)
#    ptable['OBP'] = (ptable.SNGL+ptable.XBH+ptable.HR+ptable.BB)/(ptable.SNGL+ptable.XBH+ptable.HR+ptable.K+ptable.BIPOUT+ptable.BB)
#    ptable['WOBA'] = (ptable.SNGL*0.89+ptable.XBH*1.31+ptable.HR*2.10+ptable.BB*0.70)/(1-ptable.OTHER)
#    ptable['FIP'] = (ptable.HR*13+ptable.BB*3-ptable.K*2)/(ptable.K+ptable.BIPOUT)*3+3.05
    ptable['AVG'] = (ptable.SNGL+ptable.XBH+ptable.HR)/(1-ptable.BB)
    ptable['OBP'] = ptable.SNGL+ptable.XBH+ptable.HR+ptable.BB
    c = fg.loc[year]
    ptable['WOBA'] = ptable.SNGL*c.w1B + ptable.XBH*(c.w2B*0.9+c.w3B*0.1) + ptable.HR*c.wHR + ptable.BB*(c.wBB*0.9+c.wHBP*0.1)
    ptable['FIP'] = (ptable.HR*13+ptable.BB*3-ptable.K*2)/(ptable.K+ptable.BIPOUT)*3+c.cFIP
    return ptable

# A function to get all of the matrices for any specified year; modify for tau model
# NOT DONE YET
def tau_analysis(year,alpha=0.001,tau=[1.0,1.0,1.0,1.0,1.0]):
    # Get the relevant events
    ev = get_events(year)
    ev = ev[ev.event!='OTHER']
    ev = ev[['batter','pitcher','gamesite','timesthrough','pitbathand','event']]
    ev['ind'] = 1.0
    # Calculate the mean probabilities, ratios, and logratios
    pbar = ev.event.value_counts(normalize=True).to_frame().transpose()
    pbar = pbar[['SNGL','XBH','HR','BB','K','BIPOUT']]
    rbar = pbar / (1-pbar)
    logrbar = np.log(rbar)
    # Pivot to get the indicators
    xbatter = ev.pivot(columns='batter',values='ind').fillna(0)
    xpitcher = ev.pivot(columns='pitcher',values='ind').fillna(0)
    xgamesite = ev.pivot(columns='gamesite',values='ind').fillna(0)
    xtimesthrough = ev.pivot(columns='timesthrough',values='ind').fillna(0)
    xpitbathand = ev.pivot(columns='pitbathand',values='ind').fillna(0)
    # Concatenate the indicators for the array
    xbatter.columns = pd.MultiIndex.from_product([['batter'],xbatter.columns])
    xpitcher.columns = pd.MultiIndex.from_product([['pitcher'],xpitcher.columns])
    xgamesite.columns = pd.MultiIndex.from_product([['gamesite'],xgamesite.columns])
    xtimesthrough.columns = pd.MultiIndex.from_product([['timesthrough'],xtimesthrough.columns])
    xpitbathand.columns = pd.MultiIndex.from_product([['pitbathand'],xpitbathand.columns])
    x = pd.concat([xbatter,xpitcher,xgamesite,xtimesthrough,xpitbathand],axis=1)
    x.columns.names=['split','ID']
    # Get the Y array (outcomes)
    yp = ev.pivot(columns='event',values='ind').fillna(0)
    yp = yp[['SNGL','XBH','HR','BB','K','BIPOUT']]
    yp = yp.replace(1,1-alpha)
    yp = yp.replace(0,alpha/5)
    yr = yp/(1-yp)
    ylogr = np.log(yr)
    y = np.subtract(ylogr,logrbar)
    # Solve the system
    bhat = pd.DataFrame(la.lstsq(np.matmul(x.transpose().to_numpy(),x.to_numpy()),np.matmul(x.transpose().to_numpy(),y.to_numpy()))[0])
    bhat.index = x.columns
    bhat.columns = y.columns
    rhat = np.exp(np.add(bhat,logrbar))
    phat = rhat/(1+rhat)
    return x,y,pbar,phat