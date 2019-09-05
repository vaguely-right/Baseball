library("readr")
library("dplyr")

setwd("D:/Data")

#Load in the data
dat <- read_csv("FG_Dashboard_1946-2019_Offense.csv")

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
