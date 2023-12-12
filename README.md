# PaperCSBA
Individual paper for Computer Science for Business analytics by Kasper de Koning (704733). This project is about finding duplicates in a big dataset as computationally efficient as possible. The respective duplicate detection is conducted on a tv model dataset that includes tv descriptions of multiple online retailers.

The structure of the code is as follows:

**1)** Extracting, cleaning and standardizing the key attributes in the data
**2)** Creating functions that perform minhashing to create a signature matrix
**3)** Perform LSH to find candidate pairs
**4)** Find potential duplicates through clustering
**5)** Perform bootstrapping and calculate the performance measures
**6)** Visualise the results via plots

The code can be run as is. 
