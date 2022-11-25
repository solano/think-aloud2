# Load packages -----
library(tidyverse)
library(lme4)
library(lmerTest)

# Load data -------
subvoldf <- read.csv('C:\\Users\\User\\Drive USP\\école\\2a\\p2\\modal\\python\\volume_subrows.csv',
                  sep='\t', fileEncoding='utf8')
rowvoldf <- read.csv('C:\\Users\\User\\Drive USP\\école\\2a\\p2\\modal\\python\\volume_rows.csv',
                     sep='\t', fileEncoding='utf8')
probvoldf <- read.csv('C:\\Users\\User\\Drive USP\\école\\2a\\p2\\modal\\python\\volume_probes.csv',
                     sep='\t', fileEncoding='utf8')

# Fixed effects models, subrow level  ---------
fe01 <- lm(vol ~ ADHD, data=subvoldf)
fe01c <- lm(vol ~ ADHD + age + genre, data=subvoldf)
fe02 <- lm(vol ~ ADHD_inatt + ADHD_impuls, data=subvoldf)
fe03 <- lm(vol ~ MEWS, data=subvoldf)
fe04 <- lm(vol ~ ADHD + MEWS, data=subvoldf)

# Fixed effects models, row level  ---------
fe01 <- lm(vol ~ ADHD, data=subvoldf)
fe01c <- lm(vol ~ ADHD + age + genre, data=rowvoldf)
fe02 <- lm(vol ~ ADHD_inatt + ADHD_impuls, data=rowvoldf)
fe03 <- lm(vol ~ MEWS, data=rowvoldf)
fe04 <- lm(vol ~ ADHD + MEWS, data=rowvoldf)

# Fixed effects models, probe level  ---------
fe01 <- lm(vol ~ ADHD, data=probvoldf)
fe01c <- lm(vol ~ ADHD + age + genre, data=probvoldf)
fe02 <- lm(vol ~ ADHD_inatt + ADHD_impuls, data=probvoldf)
fe03 <- lm(vol ~ MEWS, data=probvoldf)
fe04 <- lm(vol ~ ADHD + MEWS, data=probvoldf)

