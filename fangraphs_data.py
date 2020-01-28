import requests
from lxml import html
import pandas as pd

pd.set_option('display.width',150)
pd.set_option('display.max_columns',16)

#%%
#Figuring out how to extract data from a Fangraphs query
url = 'https://www.fangraphs.com/leaders.aspx?pos=all&stats=bat&lg=all&qual=0&type=0&season=2019&month=0&season1=1955&ind=0&team=0,ss&rost=0&age=0&filter=&players=0&startdate=&enddate=&page=1_10000'
xp = '//*[@id="LeaderBoard1_dg1_ctl00"]'
#xp = '//*[@id="LeaderBoard1_dg1_ctl00"]/tbody'
page = requests.get(url)
doc = html.fromstring(page.content)
table = doc.xpath(xp)[0]

colnames = [i.text_content() for i in table.xpath('//th')]

df = pd.DataFrame([],columns = colnames)
#table.xpath('//tbody')[1] is the table body
#table.xpath('//tbody')[1][N] is the Nth row in the table
#table.xpath('//tbody')[1][N][X] is the Xth entry in the Nth row
for i in table.xpath('//tbody')[1]:
    df = df.append(pd.DataFrame([[j.text_content() for j in i]],columns=colnames))

df = df.set_index('#')

                  

#%%
#Making a function to do that
def query_fangraphs(url):
    #Modify the URL to get every page of data
    url = url + '&page=1_10000'
    #Define the xpath
    xp = '//*[@id="LeaderBoard1_dg1_ctl00"]'
    #Get the page
    page = requests.get(url)
    #Get the content
    doc = html.fromstring(page.content)
    #Get the table
    table = doc.xpath(xp)[0]
    #Define the column names
    colnames = [i.text_content() for i in table.xpath('//th')]
    #Build the dataframe
    df = pd.DataFrame([],columns = colnames)
    #Extract the table data and put it in the dataframe
    for i in table.xpath('//tbody')[1]:
        df = df.append(pd.DataFrame([[j.text_content() for j in i]],columns=colnames))
    #Reset the index
    df = df.set_index('#')
    #Reformat numbers as numbers instead of text
    df = df.apply(pd.to_numeric,errors='ignore')
    return df

#%%
#Test the function a bit
url = 'https://www.fangraphs.com/leaders.aspx?pos=all&stats=bat&lg=all&qual=y&type=8&season=2019&month=0&season1=2019&ind=0'
df = query_fangraphs(url)

#%%
#Now let's try some stuff
url = 'https://www.fangraphs.com/leaders.aspx?pos=all&stats=bat&lg=all&qual=0&type=0&season=2019&month=0&season1=1955&ind=0&team=0,ss&rost=0&age=0&filter=&players=0&startdate=&enddate='
df = query_fangraphs(url)
df['xPA'] = df['1B'] + df['2B'] + df['3B'] + df.HR + df.BB + df.IBB + df.SO + df.HBP + df.SF + df.SH
df[['PA','xPA']]

#%%
#Try a bit different
url = 'https://www.fangraphs.com/leaders.aspx?pos=all&stats=bat&lg=all&qual=0&type=c,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,37,38,41,35,34&season=2019&month=0&season1=1955&ind=0&team=0,ss&rost=0&age=0&filter=&players=0&startdate=1955-01-01&enddate=2019-12-31'
df = query_fangraphs(url)

#%% Should have left the glove at home
url = 'https://www.fangraphs.com/leaders.aspx?pos=all&stats=bat&lg=all&qual=y&type=8&season=2019&month=0&season1=1973&ind=1&team=&rost=&age=&filter=&players=&startdate=&enddate='
df = query_fangraphs(url)
df = df[["Name","Season", "G", "wRC+", "Def", "WAR"]]
df['DefDH'] = -17.5 * df.G / 162
df['DefImp'] = df.DefDH - df.Def
df['DefImp162'] = df.DefImp * 162 / df.G
df.sort_values(by='DefImp',ascending=False).head(25)
df.sort_values(by='DefImp162',ascending=False).head(25)

#%% Career numbers
url = 'https://www.fangraphs.com/leaders.aspx?pos=all&stats=bat&lg=all&qual=y&type=8&season=2019&month=0&season1=1973&ind=0&team=&rost=&age=&filter=&players=&startdate=&enddate='
df = query_fangraphs(url)
df = df[["Name", "G", "wRC+", "Def", "WAR"]]
df['DefDH'] = -17.5 * df.G / 162
df['DefImp'] = df.DefDH - df.Def
df['DefImp162'] = df.DefImp * 162 / df.G
df.sort_values(by='DefImp',ascending=False).head(25)
df.sort_values(by='DefImp162',ascending=False).head(25)






