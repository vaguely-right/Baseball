library("readr")
library("dplyr")

#setwd("D:/Data")
urlfile <- "https://raw.githubusercontent.com/vaguely-right/Baseball/master/FG_Dashboard_1973-2019.csv?token=AMYNFAG3UDNY75KAUGUIGE25OB3H2"

#Look season-by-season

#To get the data as a tibble:
#dat <- read_csv("FG_Dashboard_1973-2019.csv")
dat <- read_csv(urlfile)

#To get the data as a dataframe:
#dat <- read.csv("FG_Dashboard_1973-2019.csv")
#dat <- read.csv(urlfile)

colnames(dat)
keepcols <- c("Season", "Name", "Team", "G", "wRC+", "Def", "WAR")
dat1 <- dat[keepcols]

dat1 <- arrange(dat1,Def)
dat1$DefDH <- -17.5 * dat1$G / 162
dat1$DHImp <- dat1$DefDH - dat1$Def

dat2 <- filter(dat1, DHImp>0)
dat2 <- arrange(dat2, desc(DHImp))
View(dat2)


#Look at career numbers
#dat <- read_csv("FG_Dashboard_1973-2019_sum.csv")
urlfile <- "https://raw.githubusercontent.com/vaguely-right/Baseball/master/FG_Dashboard_1973-2019_sum.csv?token=AMYNFAGSKGSYH27DZ4HYPCS5OB3RG"
dat <- read_csv(urlfile)
keepcols <- c("Name", "G", "wRC+", "Def", "WAR")
dat1 <- dat[keepcols]
dat1 <- arrange(dat1,Def)
dat1$DefDH <- -17.5 * dat1$G / 162
dat1$DHImp <- dat1$DefDH - dat1$Def
dat1$DHImp162 <- dat1$DHImp * 162 / dat1$G
dat2 <- filter(dat1, DHImp>0)
dat2 <- arrange(dat2, desc(DHImp162))
View(dat2)


#Try scraping the data from the website directly rather than a CSV file
library("textreadr")
library("rvest")
library("tibble")

url <- "https://www.fangraphs.com/leaders.aspx?pos=all&stats=bat&lg=all&qual=y&type=8&season=2019&month=0&season1=1973&ind=1&team=&rost=&age=&filter=&players=&startdate=&enddate=&page=1_2000"
xp <- '//*[@id="LeaderBoard1_dg1_ctl00"]'

dat <- url %>%
  read_html() %>%
  html_nodes(xpath=xp) %>%
  html_table()

dat1 <- dat[[1]]
dat1 <- dat1[-c(1,3),]
colnames(dat1) <- dat1[1,]
dat1 <- dat1[-c(1),]
View(dat1)

keepcols <- c("Name", "G", "wRC+", "Def", "WAR")
dat2 <- as_tibble(dat1[keepcols])
dat2$G <- as.numeric(dat2$G)
dat2$'wRC+' <- as.numeric(dat2$'wRC+')
dat2$Def <- as.numeric(dat2$Def)
dat2$WAR <- as.numeric(dat2$WAR)

dat2 <- arrange(dat2,Def)
dat2$DefDH <- -17.5 * dat2$G / 162
dat2$DHImp <- dat2$DefDH - dat2$Def
dat2$DHImp162 <- dat2$DHImp * 162 / dat2$G
dat3 <- filter(dat2, DHImp>0)
dat3 <- arrange(dat3, desc(DHImp162))
View(dat3)


#Now, scraping from the website for career numbers
library("textreadr")
library("rvest")
library("tibble")

url <- "https://www.fangraphs.com/leaders.aspx?pos=all&stats=bat&lg=all&qual=y&type=8&season=2019&month=0&season1=1973&ind=0&team=&rost=&age=&filter=&players=&startdate=&enddate=&page=1_2000"
xp <- '//*[@id="LeaderBoard1_dg1_ctl00"]'

dat <- url %>%
  read_html() %>%
  html_nodes(xpath=xp) %>%
  html_table()

dat1 <- dat[[1]]
dat1 <- dat1[-c(1,3),]
colnames(dat1) <- dat1[1,]
dat1 <- dat1[-c(1),]
View(dat1)

keepcols <- c("Name", "G", "wRC+", "Def", "WAR")
dat2 <- as_tibble(dat1[keepcols])
dat2$G <- as.numeric(dat2$G)
dat2$'wRC+' <- as.numeric(dat2$'wRC+')
dat2$Def <- as.numeric(dat2$Def)
dat2$WAR <- as.numeric(dat2$WAR)

dat2 <- arrange(dat2,Def)
dat2$DefDH <- -17.5 * dat2$G / 162
dat2$DHImp <- dat2$DefDH - dat2$Def
dat2$DHImp162 <- dat2$DHImp * 162 / dat2$G
dat3 <- filter(dat2, DHImp>0)
dat3 <- arrange(dat3, desc(DHImp162))
View(dat3)
