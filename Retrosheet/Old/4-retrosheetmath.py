import pandas as pd
import numpy as np
from tqdm import tqdm
import numpy.linalg as la
import seaborn as sns
from sklearn import linear_model as lm

pd.set_option('display.width',150)
pd.set_option('display.max_columns',16)
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


#%%
# Get the events for a specified year
cols = ['SNGL','XBH','HR','BB','K','BIPOUT']
splits = ['batter','pitcher','gamesite','timesthrough','pitbathand']
year = 2013
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
yp = yp.replace(1,0.99999)
yp = yp.replace(0,0.000002)
yr = yp/(1-yp)
ylogr = np.log(yr)
y = np.subtract(ylogr,logrbar)



#%% Look at some variances etc
ev.groupby('batter').event.value_counts(normalize=True).to_frame().rename(columns={'event':'frac'}).reset_index().pivot(index='batter',columns='event',values='frac').var()
ev.groupby('batter').event.value_counts(normalize=True).to_frame().rename(columns={'event':'frac'}).reset_index().pivot(index='batter',columns='event',values='frac').hist()
ev.groupby('pitcher').event.value_counts(normalize=True).to_frame().rename(columns={'event':'frac'}).reset_index().pivot(index='pitcher',columns='event',values='frac').hist()


ev.groupby('batter').event.value_counts(normalize=True).to_frame().rename(columns={'event':'frac'}).reset_index().pivot(index='batter',columns='event',values='frac').std().to_frame().rename(columns={0:'batter'}).transpose()
ev.groupby('pitcher').event.value_counts(normalize=True).to_frame().rename(columns={'event':'frac'}).reset_index().pivot(index='pitcher',columns='event',values='frac').std()
ev.groupby('gamesite').event.value_counts(normalize=True).to_frame().rename(columns={'event':'frac'}).reset_index().pivot(index='gamesite',columns='event',values='frac').std()
ev.groupby('timesthrough').event.value_counts(normalize=True).to_frame().rename(columns={'event':'frac'}).reset_index().pivot(index='timesthrough',columns='event',values='frac').std()
ev.groupby('pitbathand').event.value_counts(normalize=True).to_frame().rename(columns={'event':'frac'}).reset_index().pivot(index='pitbathand',columns='event',values='frac').std()


sd = []
for s in splits:
    sd.append(ev.groupby(s).event.value_counts(normalize=True).to_frame().rename(columns={'event':'frac'}).reset_index().pivot(index=s,columns='event',values='frac').std().to_frame().rename(columns={0:s}).transpose())

pd.concat(sd)

#%%
# Try to solve the system using linear algebra
bhat = np.matmul(np.matmul(la.inv(np.matmul(x.transpose().to_numpy(),x.to_numpy())),x.transpose().to_numpy()),y.to_numpy())

# Singular matrix error, try the solver ax=b
bhat = la.solve(np.matmul(x.transpose().to_numpy(),x.to_numpy()),np.matmul(x.transpose().to_numpy(),y.to_numpy()))

# Singular matrix error, let's look at the matrix rank
lhs = np.matmul(x.transpose().to_numpy(),x.to_numpy())
np.shape(lhs)
la.matrix_rank(lhs)
# Rank deficient. Size is 1659, rank 1653.

#%%
#Try to solve the system using matrix math
bhat = pd.DataFrame(la.lstsq(np.matmul(x.transpose().to_numpy(),x.to_numpy()),np.matmul(x.transpose().to_numpy(),y.to_numpy()))[0])
bhat.index = x.columns
bhat.columns = y.columns

#Hey, that worked! And it looks like it may have found the pseudoinverse, saving me the trouble

#%%
# Try to solve the system using scikit-learn
reg = lm.LinearRegression(fit_intercept=False)
reg.fit(x.to_numpy(),y.to_numpy())

bhat = pd.DataFrame(reg.coef_.transpose())
bhat.index = x.columns
bhat.columns = y.columns


# WTF. Values are enormous. Problem with rank-deficient?


#%%
# Try ridge regression
reg = lm.Ridge(alpha=100.0,fit_intercept=False)
reg.fit(x.to_numpy(),y.to_numpy())

bhat = pd.DataFrame(reg.coef_.transpose())
bhat.index = x.columns
bhat.columns = y.columns

# Works to constrain the values, but same problem with unbalanced sums

#%% Try lasso regression
reg = lm.Lasso(alpha=0.1,fit_intercept=False)
reg.fit(x.to_numpy(),y.to_numpy())

bhat = pd.DataFrame(reg.coef_.transpose())
bhat.index = x.columns
bhat.columns = y.columns

#%%
# Look a the splits and histograms
bhat.groupby('split').mean()
bhat.groupby('split').std()
bhat.loc['batter'].hist()
bhat.loc['gamesite'].hist()
bhat.loc['pitbathand'].hist()
bhat.loc['pitcher'].hist()
bhat.loc['timesthrough'].hist()

#%%

# Take the bhat estimate and put it back in probability space
rhat = np.exp(np.add(bhat,logrbar))

rhat.groupby('split').mean()

# Okay, now get the original probabilities
phat = rhat/(1+rhat)
phat.groupby('split').mean()
phat.loc['batter'].hist()
phat.loc['pitcher'].hist()
[phat.loc[s].hist() for s in splits]

phat['SUM'] = np.sum(phat,axis=1)
phat.groupby('split').mean()

#%% Some checks

a = []
for s in splits:
    a.append(x[s].dot(phat.loc[s]).mean())

a

x.dot(bhat).mean()
x['batter'].dot(bhat.loc['batter']).mean()

np.mean(np.matmul(x['batter'].to_numpy(),phat.loc['batter'].to_numpy()),axis=0)
x['batter'].dot(phat.loc['batter']).mean()
x['pitcher'].dot(phat.loc['pitcher']).mean()
x['pitbathand'].dot(phat.loc['pitbathand']).mean()

np.mean(np.matmul(x['pitcher'].to_numpy(),phat.loc['pitcher'].to_numpy()),axis=0)

np.mean(np.matmul(x['gamesite'].to_numpy(),phat.loc['gamesite'].to_numpy()),axis=0)

np.mean(np.matmul(x['timesthrough'].to_numpy(),phat.loc['timesthrough'].to_numpy()),axis=0)

np.mean(np.matmul(x['pitbathand'].to_numpy(),phat.loc['pitbathand'].to_numpy()),axis=0)

#Don't sum to 1.0; normalize for now
phat = np.divide(phat,np.sum(phat,axis=1).to_frame())

#%%
# Get some of the real values to compare
idfile = 'retroID.csv'
pid = pd.read_csv(idfile,index_col=False)
pid['Name'] = pid.First+' '+pid.Last
pid = pid[['ID','Name']]
pid.set_index('ID',inplace=True)

bat = pivot_events(year,'batter')
#bat = bat.merge(pid[['Name']],how='left',left_on='batter',right_on='ID')
bat = bat.merge(pid,left_index=True,right_index=True)
bat = bat[['Name','PA','SNGL','XBH','HR','BB','K','BIPOUT','AVG','OBP','WOBA','FIP']]

bathat = phat.loc['batter']
bathat['AVG'] = (bathat.SNGL+bathat.XBH+bathat.HR)/(1-bathat.BB)
bathat['OBP'] = bathat.SNGL+bathat.XBH+bathat.HR+bathat.BB
c = fg.loc[year]
bathat['WOBA'] = bathat.SNGL*c.w1B + bathat.XBH*(c.w2B*0.9+c.w3B*0.1) + bathat.HR*c.wHR + bathat.BB*(c.wBB*0.9+c.wHBP*0.1)
bathat['FIP'] = (bathat.HR*13+bathat.BB*3-bathat.K*2)/(bathat.K+bathat.BIPOUT)*3+c.cFIP
bathat = bathat.merge(pid,on='ID',how='left')
bathat = bathat.merge(bat[['ID','PA']],how='left',on='ID')
bathat = bathat[['ID','Name','PA','SNGL','XBH','HR','BB','K','BIPOUT','AVG','OBP','WOBA','FIP']]

bat[bat.PA>=500].sort_values('WOBA',ascending=False).head(10)
bathat[bathat.PA>=500].sort_values('WOBA',ascending=False).head(10)

sns.scatterplot(x=bat[bat.PA>=500].WOBA,y=bathat[bathat.PA>=500].WOBA)

# It's correlated but the overall scale seems off

#%%
# Pitchers now
pit = pivot_events(year,'pitcher')
pit = pit.merge(pid[['ID','Name']],how='left',left_on='pitcher',right_on='ID')
pit = pit[['ID','Name','PA','SNGL','XBH','HR','BB','K','BIPOUT','AVG','OBP','WOBA','FIP']]

pithat = phat.loc['pitcher']
pithat['AVG'] = (pithat.SNGL+pithat.XBH+pithat.HR)/(1-pithat.BB)
pithat['OBP'] = pithat.SNGL+pithat.XBH+pithat.HR+pithat.BB
c = fg.loc[year]
pithat['WOBA'] = pithat.SNGL*c.w1B + pithat.XBH*(c.w2B*0.9+c.w3B*0.1) + pithat.HR*c.wHR + pithat.BB*(c.wBB*0.9+c.wHBP*0.1)
pithat['FIP'] = (pithat.HR*13+pithat.BB*3-pithat.K*2)/(pithat.K+pithat.BIPOUT)*3+c.cFIP
pithat = pithat.merge(pid,on='ID',how='left')
pithat = pithat.merge(pit[['ID','PA']],how='left',on='ID')
pithat = pithat[['ID','Name','PA','SNGL','XBH','HR','BB','K','BIPOUT','AVG','OBP','WOBA','FIP']]

pit[pit.PA>=500].sort_values('FIP').head(10)
pithat[pithat.PA>=500].sort_values('FIP').head(10)

sns.scatterplot(x=pit[pit.PA>=500].FIP,y=pithat[pithat.PA>=500].FIP)

#%%
# Stadiums
site = pivot_events(year,'gamesite')
idfile = 'parkcode.csv'
pid = pd.read_csv(idfile,index_col=False)
pid.set_index('PARKID',inplace=True)
site = site.merge(pid[['NAME']],how='left',left_index=True,right_index=True)
site = site[['NAME','SNGL','XBH','HR','BB','K','BIPOUT','WOBA','FIP']]

sitehat = phat.loc['gamesite']
c = fg.loc[year]
sitehat['WOBA'] = sitehat.SNGL*c.w1B + sitehat.XBH*(c.w2B*0.9+c.w3B*0.1) + sitehat.HR*c.wHR + sitehat.BB*(c.wBB*0.9+c.wHBP*0.1)
sitehat['FIP'] = (sitehat.HR*13+sitehat.BB*3-sitehat.K*2)/(sitehat.K+sitehat.BIPOUT)*3+c.cFIP
sitehat = sitehat.merge(pid[['NAME']],how='left',left_index=True,right_index=True)
sitehat = sitehat[['NAME','SNGL','XBH','HR','BB','K','BIPOUT','WOBA','FIP']]

site.sort_values('FIP').head()
sitehat.sort_values('FIP').head()

sns.scatterplot(x=site.FIP,y=sitehat.FIP)
sns.scatterplot(x=site.WOBA,y=sitehat.WOBA)
sns.scatterplot(x=site.HR,y=sitehat.HR)
sns.scatterplot(x=site.BIPOUT,y=sitehat.BIPOUT)









