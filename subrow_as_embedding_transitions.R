# Load packages -----
library(tidyverse)
library(lme4)
library(lmerTest)

# Load data -------
subdf <- read.csv('C:\\Users\\User\\Drive USP\\école\\2a\\p2\\modal\\python\\subrow_as_embedding_transitions.csv',
                   sep='\t', fileEncoding='utf8')

# Standardize data for analysis -------

c.length <- subdf$length - 0.5                # between -0.5 and 0.5
c.age <- (subdf$age - mean(subdf$age[!is.na(subdf$age)])) # in years
c.ADHD <- (subdf$ADHD - mean(subdf$ADHD))/5   # in units of 5 ADHD points
c.ADHD_inatt <- (subdf$ADHD_inatt - mean(subdf$ADHD_inatt))/5
c.ADHD_impuls <- (subdf$ADHD_impuls - mean(subdf$ADHD_impuls))/5
c.MEWS <- (subdf$MEWS - mean(subdf$MEWS))/5 # units of 5 MEWS points

csubdf = data.frame(suj = subdf$suj,
                     bloc = subdf$bloc,
                     prob = subdf$prob,
                     idprob = 1000*subdf$suj + 100*subdf$bloc + subdf$prob,
                     length = c.length,
                     age = c.age,
                     genre = subdf$genre,
                     exp = subdf$exp,
                     level = subdf$level,
                     topic = subdf$topic,
                     ADHD = c.ADHD,
                     ADHD_inatt = c.ADHD_inatt,
                     ADHD_impuls = c.ADHD_impuls,
                     MEWS = c.MEWS)

# Fixed effects models ------

fe01 <- lm(length ~ ADHD, data=csubdf)

# Mixed effects models ------

me01 <- lmer(length ~ ADHD + (1|suj), data=csubdf)
me02 <- lmer(length ~ ADHD + genre + age + (1|suj), data=csubdf)

me03 <- lmer(ADHD ~ length + (1|suj), data=csubdf) # does not converge

me04 <- lmer(length ~ ADHD_inatt + ADHD_impuls + (1|suj), data=csubdf)
me05 <- lmer(length ~ ADHD_inatt + ADHD_impuls + genre + age + (1|suj), data=csubdf)

# MEWS does not predict anything
me06 <- lmer(length ~ ADHD + MEWS + (1|suj), data=csubdf)
me07 <- lmer(length ~ ADHD + MEWS + age + genre + (1|suj), data=csubdf)
me08 <- lmer(length ~ ADHD_inatt + ADHD_impuls + MEWS + (1|suj), data=csubdf)
me09 <- lmer(length ~ ADHD_inatt + ADHD_impuls + MEWS + age + genre + (1|suj), data=csubdf)

# Add probe-level predictor
me10 <- lmer(length ~ ADHD + (1|suj) + (1|idprob), data=csubdf)
# TODO: see how manyy unique probes there are

me11 <- lmer(length ~ MEWS + (1|suj), data=csubdf)

# ----------
# We want to:
# 1. "Keep it maximal", i.e. fit the most complex model consistent with
# experimental design that does not result in a singular fit
# 2. Eventually compare models through some criterion like AIC or BIC
