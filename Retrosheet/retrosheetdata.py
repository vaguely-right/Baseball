import pandas as pd

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


