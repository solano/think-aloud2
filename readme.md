This code analyses data from the "Think aloud" experiment. It is meant for use by the experimentators.


Structure of the analysis
-------------------------

The general idea is that people's (reported) spontaneous thought follows a random walk in semantic space, see references [5, 6]. We speak of a semantic "space" because there is a notion of distance between meanings (dog is near cat and far from spoon). This distance is measured as a cosine distance between vectors of a phrase embedding performed using the LASER model.

Transcribed data is analysed in three different levels of segmentation:
- The "probe" level, in which the whole text of each probe is embedded as a single point, and a "block" forms a trajectory of a few points.

- The "row" level, the natural segmentation of the data, where each row of text (segment of speech separated from others by a pause of >500ms) is embedded as a single point, and each probe forms a trajectory containing usually more than a dozen points.

- The "subrow" level, where even the text in each row is divided into several phrases according to the oral marker of pauses "(.)". Each of these phrases (often consisting of a single word) is embedded into a single point, and each probe forms a trajectory containing dozens of points.

The different measurements performed
------------------------------------

The following quantities of interest ("indices") are computed in the code.

- *Length* refers to the cosine distance between successive points in a trajectory.

- *Interval* refers to the difference between the middle times (halfway between the endtime and the starting time) of successive points. Only available at row level.

- *Pause* refers to the difference between the endtime of one point and the starting time of the following point. Only available at row level.

- *Speed* refers to length divided by interval.

- *Volume* refers to a measure of size of a point cloud associated to each subject. It is defined as the geometric mean of the semiaxes of the Minimum Volume Enclosing Ellipsoid (MVEE) of this cloud of points, cf. ref. [3]. At each level of analysis, there is one volume per subject.

- *Circuitousness* refers to a measure associated to each trajectory. It is defined as the total length of the trajectory (sum of lengths of jumps) divided by the minimum length possible for a trajectory passing through the same points, once through each point, and starting and ending at the same points, cf. ref. [3]. At each level of analysis, there is one circuitousness per trajectory.

Stylised facts
--------------

The ADHD questionnaire has been divided into an inattentiveness score and an impulsivity score, which summed together form the (total) ADHD score. Hereafter I call those simply "inattentiveness", "impulsivity" and "ADHD").

- Inattention increases volume, impulsivity decreases it (they cancel out so that ADHD has no effect).
- Impulsivity increases circuitousness, inattention has no effect on it.
- Sex and age predict nothing.
- MEWS is very correlated with ADHD.

At probe level:
- Impulsivity decreases length.

At row level:
- Inattention decreases speed, has no effect on length and increases interval.
- Impulsivity has no effect on speed, length or interval.

At subrow level:
- There is no effect.

Attempt at interpretation of the difference between segmentation levels: LASER works best for longer phrases, but the longer the phrase we give it, the less data points we have.

File structure
--------------

Python code for preprocessing, embedding and analysing data is to be found in `think_aloud/'. Data is (predictably) in `data/' and some pytests are in `tests/', used to ensure that `compute_indices.py' works properly. Starting from the raw data file `data_stream_text_time.csv', one must execute, in this order:

- `preprocess.py' to clean up oral markers from transcribed speech and run a spellchecker (this generates the data files `text_*.csv'. The original structure of the dataset in kept in `data_stream_text_time_clean.csv')

- `embed.py' to embed the generated text using LASER (this generates the data files `*_embeddings.npy');

- notebooks/compute_indices.ipynb to compute quantities on embedded data and export data files for statistical analysis (this generates all other data files);

- All other files in notebooks/, which are R notebooks used to do statistical tests with the package `lmerTest'.


Warnings
--------

- It's been a year since I ran preprocessing and embedding; code may require some changes now (especially prefixing filenames with correct directories).
- If you want to redo preprocessing, it is not guaranteed to produce the same output now as was produced back then because it runs a third-party spellchecker that may have been changed.
- Original code was run in blocks through IPython, which allowed for a rudimentary notebook feel and explains the Python comments starting with "%%". Order of execution problems may mean you want to transform python files into Jupyter notebooks as I did with compute_indices.py. A similar remark holds for the R files (comments ending in "---"), but those are already transformed into notebooks.

Dependencies
------------

- LASER: https://github.com/yannvgn/laserembeddings https://github.com/facebookresearch/LASER and its dependencies;
- subword-nmt (v.0.3.7, later versions seem not to work), a particular dependency of LASER;
- languagetool, used for spellchecking.

References
----------

1. Gray, K. et al. “Forward flow”: A new measure to quantify free thought and predict creativity. American Psychologist 74, 539–554 (2019).
2. Heusser, A. C., Fitzpatrick, P. C. & Manning, J. R. Geometric models reveal behavioural and neural signatures of transforming experiences into memories. Nat Hum Behav 5, 905–919 (2021).
3. Toubia, O., Berger, J. & Eliashberg, J. How quantifying the shape of stories predicts their success. Proc. Natl. Acad. Sci. U.S.A. 118, e2011695118 (2021).
4. Van den Driessche, C., Chevrier, F., Cleeremans, A. & Sackur, J. Lower Attentional Skills predict increased exploratory foraging patterns. Sci Rep 9, 10948 (2019).
5. Elias Costa, M. Scale-invariant transition probabilities in free word association trajectories. Front. Integr. Neurosci. 3, (2009).
6. Mildner, J. N. & Tamir, D. I. Spontaneous Thought as an Unconstrained Memory Process. Trends in Neurosciences 42, 763–777 (2019).
