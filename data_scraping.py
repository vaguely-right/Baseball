import pandas as pd
import numpy as np
from baseball_scraper import statcast

pd.options.display.max_columns = 16
pd.options.display.width = 156


#%%
data = statcast('2019-02-01','2019-04-23')
sum(data.events.notnull())

data.columns
df = data[data.events.notnull()][['index','game_date','batter','pitcher','events','stand','p_throws','home_team','away_team']].copy()

