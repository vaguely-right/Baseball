library("readr")
library("dplyr")

#setwd("D:/Data")

#Load in the data
#dat <- read_csv("FG_Dashboard_1946-2019_Offense.csv")
urlfile <- "https://raw.githubusercontent.com/vaguely-right/Baseball/master/FG_Dashboard_1946-2019_Offense.csv?token=AMYNFAACD7NSCH3QC4YOKBK5OB4GC"
dat <- read_csv(urlfile)

#Scrape the data directly from the website
library("textreadr")
library("rvest")
library("tibble")
url <- "https://www.fangraphs.com/leaders.aspx?pos=all&stats=bat&lg=all&qual=y&type=c,4,6,11,12,13,21,-1,34,35,40,41,-1,23,37,38,50,61,52,51,53,-1,111,-1,203,199,58&season=2019&month=0&season1=1946&ind=1&team=&rost=&age=&filter=&players=&startdate=&enddate=&page=1_10000"
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

dat <- as_tibble(dat1)
dat$RBI <- as.numeric(dat$RBI)
dat$wRC <- as.numeric(dat$wRC)
dat$wRAA <- as.numeric(dat$wRAA)
dat$BsR <- as.numeric(dat$BsR)
dat$Bat <- as.numeric(dat$Bat)

#Get the columns that we want
colnames(dat)
keepcols <- c("Season", "Name", "Team", "RBI", "wRC", "wRAA", "BsR", "Bat", "Off", "WAR")
dat1 <- dat[keepcols]

#Look at the data a bit
dat1 <- arrange(dat1,desc(Bat))
View(dat1)

#Calculate different measures of overratedness
dat1$wRC_RBI <- dat1$wRC - dat1$RBI       
dat1$wRAA_RBI <- dat1$wRAA - dat1$RBI
dat1$Bat_RBI <- dat1$Bat - dat1$RBI
dat1$Off_RBI <- dat1$Off - dat1$RBI

#Get Batting Runs - RBIs (park and league adjusted, centered on zero)
dat2 <- arrange(dat1,Bat_RBI)
dat2 <- head(dat2,20)
View(dat2)

#Get Weighted Runs Created - RBIs (not adjusted, raw RC, counting stat)
dat3 <- arrange(dat1,wRC_RBI)
dat3 <- head(dat3,20)
View(dat3)
