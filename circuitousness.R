# Load packages -----
library(tidyverse)
library(lme4)
library(lmerTest)

# Load data -------
subcircdf <- read.csv('C:\\Users\\User\\Drive USP\\école\\2a\\p2\\modal\\python\\circ_subrows.csv',
                     sep='\t', fileEncoding='utf8')
rowcircdf <- read.csv('C:\\Users\\User\\Drive USP\\école\\2a\\p2\\modal\\python\\circ_rows.csv',
                     sep='\t', fileEncoding='utf8')
probcircdf <- read.csv('C:\\Users\\User\\Drive USP\\école\\2a\\p2\\modal\\python\\circ_probes.csv',
                      sep='\t', fileEncoding='utf8')

# Fixed effects models, subrow level  ---------
fe01 <- lm(circ ~ ADHD, data=subcircdf)
fe01c <- lm(circ ~ ADHD + age + genre, data=subcircdf)
fe02 <- lm(circ ~ ADHD_inatt + ADHD_impuls, data=subcircdf)
fe03 <- lm(circ ~ MEWS, data=subcircdf)
fe04 <- lm(circ ~ ADHD + MEWS, data=subcircdf)

# Mixed effects models, subrow level --------
me01 <- lmer(circ ~ ADHD + (1|suj), data=subcircdf)
me02 <- lmer(circ ~ ADHD_inatt + ADHD_impuls + (1|suj), data=subcircdf)
me02c <- lmer(circ ~ ADHD_inatt + ADHD_impuls + (1|suj) + age + genre, data=subcircdf)
me03 <- lmer(circ ~ MEWS + (1|suj), data=subcircdf)
me04 <- lmer(circ ~ ADHD + MEWS + (1|suj), data=subcircdf)
me05 <- lmer(circ ~ ADHD_inatt + ADHD_impuls + MEWS + (1|suj), data=subcircdf)
me05c <- lmer(circ ~ ADHD_inatt + ADHD_impuls + MEWS + (1|suj) + age + genre, data=subcircdf)

# Fixed effects models, row level  ---------
fe01 <- lm(circ ~ ADHD, data=rowcircdf)
fe01c <- lm(circ ~ ADHD + age + genre, data=rowcircdf)
fe02 <- lm(circ ~ ADHD_inatt + ADHD_impuls, data=rowcircdf)
fe03 <- lm(circ ~ MEWS, data=rowcircdf)
fe04 <- lm(circ ~ ADHD + MEWS, data=rowcircdf)

# Mixed effects models, row level --------
me01 <- lmer(circ ~ ADHD + (1|suj), data=rowcircdf)
me02 <- lmer(circ ~ ADHD_inatt + ADHD_impuls + (1|suj), data=rowcircdf)
me02c <- lmer(circ ~ ADHD_inatt + ADHD_impuls + (1|suj) + age + genre, data=rowcircdf)
me03 <- lmer(circ ~ MEWS + (1|suj), data=rowcircdf)
me04 <- lmer(circ ~ ADHD + MEWS + (1|suj), data=rowcircdf)
me05 <- lmer(circ ~ ADHD_inatt + ADHD_impuls + MEWS + (1|suj), data=rowcircdf)
me05c <- lmer(circ ~ ADHD_inatt + ADHD_impuls + MEWS + (1|suj) + age + genre, data=rowcircdf)


# Fixed effects models, probe level  ---------
fe01 <- lm(circ ~ ADHD, data=probcircdf)
fe01c <- lm(circ ~ ADHD + age + genre, data=probcircdf)
fe02 <- lm(circ ~ ADHD_inatt + ADHD_impuls, data=probcircdf)
fe03 <- lm(circ ~ MEWS, data=probcircdf)
fe04 <- lm(circ ~ ADHD + MEWS, data=probcircdf)

# Mixed effects models, probe level --------
me01 <- lmer(circ ~ ADHD + (1|suj), data=probcircdf)
me02 <- lmer(circ ~ ADHD_inatt + ADHD_impuls + (1|suj), data=probcircdf)
me02c <- lmer(circ ~ ADHD_inatt + ADHD_impuls + (1|suj) + age + genre, data=probcircdf)
me03 <- lmer(circ ~ MEWS + (1|suj), data=probcircdf)
me04 <- lmer(circ ~ ADHD + MEWS + (1|suj), data=probcircdf)
me05 <- lmer(circ ~ ADHD_inatt + ADHD_impuls + MEWS + (1|suj), data=probcircdf)
me05c <- lmer(circ ~ ADHD_inatt + ADHD_impuls + MEWS + (1|suj) + age + genre, data=probcircdf)

