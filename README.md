# Outlier Detection
<img src="https://images.ctfassets.net/kj4bmrik9d6o/cESitsCxzL2ijivbNwiW6/de9421d4c144e5a5c23c35941931c83f/Outlier_Graph_CalculateOutlierFormula-01.png">

Outlier Detection Python is a specialized task which has various use-cases in Machine
Learning. Use-cases would be anomaly detection, fraud detection, outlier detection etc. There
are many ways we can find outliers in your analysis.


## What are outliers?
Outliers: in simple terms outliers are data points which are significantly different from your
entire datasets.

## How do outliers occur in a datasets?

Outliers occur either by chance, or either by measurement error or data population is heavy
tailed distribution as shown above.
Main effects of having outliers are that they can skew your analytics in poor analysis, longer
training time and bad results at the end. Most importantly, this distorts the reality which
exists in the data.

## Simple methods to Identify outliers in your datasets.

**Sorting** 
<br>If you have dataset you can quickly just sort ascending or descending.
While it is looks so obvious, but sorting actually works on real world.<br>
**Outlier Detection Python** 
<br>Quick Method in Pandas – Describe( )

```
import numpy as np
import pandas as pd
url =
'https://raw.githubusercontent.com/Sketchjar/MachineLearningHD'
df = pd.read_csv(url)
df.describe()
```

## Outlier Detection Using Machine Learning
Robust Covariance – Elliptic Envelope<br>
This method is based on premises that outliers in a data leads increase in covariance, making
the range of data larger. Subsequently the determinant of covariance will also increase, this
in theory should reduce by removing the outliers in the datasets. This method assumes that
some of hyper parameters in n samples follow Gaussian distribution. Here is flow on how
this works:
<br>
 #### One-Class SVM
One class Support Vector Machine is a special case in support vector machines which is used
for unsupervised outlier detection. For more information on support vector

Let see outlier detection python code using One Class SVM. We will see two different
examples for it.
```
from sklearn.svm import OneClassSVM
X = [[0], [0.44], [0.45], [0.46], [1]]
clf = OneClassSVM(gamma='auto').fit(X)
clf.predict(X)
array([-1, 1, 1, 1, -1, -1, -1], dtype=int64)
```
<br>output:
```
array([-1, 1, 1, 1, -1, -1, -1], dtype=int64)
```
Here -1 refers to outlier and 1 refers to not an outliers.
<br>
#### Isolation Forest
Isolation Forest is an ensemble model which isolates observations by randomly selecting a
feature and selecting a split value between maximum and minimum of selected feature.<br>
Since this recursive partitioning is represented by a tree structure, and number of splittings is
equivalent to path length from root node to terminating node.
<br><br>

<img src="https://miro.medium.com/max/1400/1*6GFMewU1Aax57nsW2uSakQ.png" high=450 width=550>

See Isolation Forest in code.
```
from sklearn.ensemble import IsolationForest
X = [[-1.1], [0.3], [0.5], [100]]
clf = IsolationForest(random_state=0).fit(X)
clf.predict([[0.1], [0], [90]])
```
<br>output:
```
array([ 1, 1, -1])
```
Here -1 refers to outlier and 1 refers to not an outliers.
