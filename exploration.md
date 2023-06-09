# Exploration

## Table of Contents

1. [Overview](##1.-Overview)
2. [Tools](##2.-Tools)
3. [Exploration](##3.-Exploration)
    - [Extract](###3.1-Extract)
    - [Transform](###3.2-Transform)
    - [Load](###3.3-Load)
4. Evaluation Analysis
    - [Confusion Matrix](###4.1-Confusion-Matrix)
    - Precision
    - Recall
    - F1 Score
    - Accuracy
    - Specificity
    - Autoevaluation
5. [Building a Pipeline](##5.-Building-a-Pipeline)
6. [Tests](##6.-Tests)
7. [Conclusion](##7.-Conclusion)

## 1. Overview

**Task:**

The labellers have reviewed the output of a model, and now we would like to compare its performance against the ground truth. Attached are the results of the model run on 752 images, as well as the labels generated by the labellers. Each file lists the predictions of a single microscope image, with the filename denoting the image id. We would now like to compare the difference between the two datasets. Please write an efficient code base to output the F1 Score, Accuracy, Precision and Confusion Matrix of the dataset.

**Inputs:**
- Zip Folder
- Folder titled "predictions"
- JSON file describing predicted bounding boxes for a particular microscope image.
- Folder titled "Ground Truth"
- JSON file describing ground truth bounding boxes for a particular microscope image.

**Outputs:**
- F1 Score
- Accuracy
- Precision
- Confusion Matrix
- Code used to generate the outputs

**Submission**
There are two ways to submit your code:
1. Upload your repo to Github. Add ‘BioScout-Tom’ and ‘eltuna9’ as viewers to your repo.
2. Use git bundle to create a single file of your repo, and reply to this email with the bundle attached.

**Expectations:**
- When using a Jupyter notebook, please use markdown to clearly document each section.
- When using raw python, please make sure to commit frequently with clear commit messages, and to send through the bundled code.
- Bonus: Extra points for any other metrics that you believe would be useful for measuring the performance of the model.

**Goals:**
In this question, we hope for you to demonstrate your knowledge of Python applied to a common ML data wrangling task, as well as modern software development practices.

## 2. Tools

To get started, we'll begin by importing a few packages that will help us carry out the tasks above. The two packages not in the standard library are:
- `pandas`
- `scikit-learn`
- `ibis`
- `dvc`
- `OmegaConf`
- `ruff`
- `mypy`
- `pandas-stubs`

The following packages are used in the app version of this project only.
- `streamlit`
- `graphviz`
- `seaborn`


```python
import pandas as pd
import json
from pprint import pprint
from glob import glob
from os.path import join
from random import choice
from sklearn import metrics as m
import ibis
from omegaconf import OmegaConf
```

## 3. Exploration

Let's begin by grabbing all of the files in each directory and evaluating a random sample.


```python
actuals = glob(join("data", "raw", "ground_truth", "*.json"))
predicted = glob(join("data", "raw", "predictions", "*.json"))
predicted[:4]
```


```python
random_sample = choice(actuals)
random_sample
```


```python
with open(random_sample, "r") as sample:
    data = json.load(sample)
pprint(data)
```

A much nicer way to look at this file would be through the [JSON Crack](https://jsoncrack.com/editor) editor, which would give us the following image.


<img title="" src="https://raw.githubusercontent.com/ramonpzg/agri_challenge/5cb9da51a1a8e0d3b47e3161b43c2a1a3ffcdf01/images/jsonstruc.svg" alt="AJKFEHASIEFHAUISDHFwrfgasfdgsdf" data-align="center" width="600">

Let's look at the prediction for the same image.


```python
# first we'll get the same file name and connect it to its respective directory
random_prediction = join("data", "raw", "predictions", random_sample.split("/")[-1])

with open(random_prediction, "r") as sample:
    data = json.load(sample)
pprint(data)
```

If we go to the [`schema_ref` url](https://darwin-public.s3.eu-west-1.amazonaws.com/darwin_json_2_0.schema.json) that comes up in the JSON output above, we can find more information about the files we are dealing with. Nonetheless, while the classification task at hand is one with bounding boxes rather than a **"yes"**, there are infectious spores in this sample, or **"no"**, there aren't any, we can still treat it as a binary classification task by using the absence of, or the lack thereof, the annotated boxes.

Let's see how many samples lack an annotation for one or more bounding boxex from the ground truth sample.


```python
box = 0
no_box = 0
not_sure = 0
good_plants = []

for file in actuals:
    with open(file, "r") as s:
        ac = json.load(s)
        if ac["annotations"]:
            box += 1
        elif ac["annotations"] == []:
            no_box += 1
            good_plants.append(file)
        else:
            not_sure += 1
            good_plants.append(file)
print(f"The ground truth sample has {box} images with at least one instance of a decease, {no_box} are healthy, and {not_sure} are unclear!")
```


```python
with open(choice(good_plants), "r") as s:
    pprint(json.load(s))
```


```python
for file in good_plants:
    with open(file, "r") as s:
        pprint(json.load(s)["annotations"])
```

Now that we know a bit about the files we're dealing with, let's create a few functions to programmatically extract, transform, load, and evaluate all samples available as well as future ones.

Since we'll need to save files here and there throughout the steps we'll take in this notebook, we want to start with a straightforward load function that takes in a dataframe and saves it as a parquet file.

Here, we take advantage of the ipython magic command `%%writefile`, which allows turn into a script everything contained in a cell. We will use this command a few more times throughout the notebook.


```python
%%writefile src/load.py
from pathlib import Path
import pandas as pd

def save_data(data, path_out, file_name):
    path_out = Path(path_out)
    if not path_out.exists(): path_out.mkdir(parents=True)
    data.to_parquet(path_out.joinpath(file_name))
    print(f"Successfully loaded the {file_name} table!")
```

### 3.1 Extract

Next, we'll need two functions:
- One that collects all JSON files inside a directory
- Another that
    1. reads each of these files
    2. extracts the the `name` and `prediction` from it
    3. creates a dataframe for each sample and puts them in a list
    4. creates one dataframe for all samples

While there are quite a few pieces that could be separated in the last function, the goal is a single one, to create a dataframe the ground truth and the predicted data.


```python
def get_files(directory):
    return glob(join(directory, "*.json"))


def read_files(data_files: str) -> pd.DataFrame:
        
    dfs_list = []
    
    for file in data_files:    
        with open(file, "r") as sample:
            data = json.load(sample)

        item = data["item"]['name']

        if data["annotations"]:
            anno_name = data["annotations"][0]["name"]
        else:
            anno_name = "Undetected"
        
        df = pd.DataFrame(data=[[item, anno_name]], columns=["item_id", "class"])
        dfs_list.append(df)
    
    return pd.concat(dfs_list, axis=0)
```

Let's make sure our functions work.


```python
actuals = get_files("data/raw/ground_truth/")
actuals[:5]
```


```python
read_files(actuals).head(10)
```

Excellent! We'll save the function in our src directory and start preparing a package in case we come up with new functions for extracting data, or in case we want to update our current ones.

In this file called `extract.py`, we'll add one more function called `load_table` since we'll need to load data in subsequent steps. In addition, we'll go ahead and call the functions in the order above and save the files into a directory called `interim`.


```python
%%writefile src/extract.py

import pandas as pd
from os.path import join
from glob import glob
from load import save_data
import json

def get_files(directory):
    return glob(join(directory, "*.json"))


def read_files(data_files: str) -> pd.DataFrame:
        
    dfs_list = []
    
    for file in data_files:    
        with open(file, "r") as sample:
            data = json.load(sample)

        item = data["item"]['name']

        if data["annotations"]:
            anno_name = data["annotations"][0]["name"]
        else:
            anno_name = "Undetected"
        
        df = pd.DataFrame(data=[[item, anno_name]], columns=["item_id", "class"])
        dfs_list.append(df)
    
    return pd.concat(dfs_list, axis=0)

def load_table(data_path, file_name):
    return pd.read_parquet(join(data_path, file_name))

if __name__ == "__main__":
    actuals = get_files("data/raw/ground_truth/")
    predictions = get_files("data/raw/predictions/")
    df_truth = read_files(actuals)
    df_preds = read_files(predictions)
    save_data(df_truth, join("data", "interim"), "actuals_table.parquet")
    save_data(df_preds, join("data", "interim"), "predicted_table.parquet")
```

Let's test our script to make sure it works well.


```python
!python src/extract.py
```


```python
df_truth = pd.read_parquet(join("data", "interim", "actuals_table.parquet"))
df_truth.head()
```


```python
df_preds = pd.read_parquet(join("data", "interim", "predicted_table.parquet"))
df_preds.tail()
```

### 3.2 Transform

The transform stage for this project will be very straightforward as what would be most helpful here is to combine both dataset to have the ground truth labes and the predicted ones, in two adjacent columns.

We will create one function for this but note that this part of the process could be much more involved.


```python
def merge_truth_preds(df1, df2, **kwargs):
    return pd.merge(left=df1, right=df2, **kwargs)
```


```python
df = merge_truth_preds(df_truth, df_preds, left_on="item_id", right_on="item_id", suffixes=("_truth", "_pred"))
df.tail()
```


```python
df.class_truth.value_counts()
```


```python
df.class_pred.value_counts()
```

Time to create a script to automate the process. We'll follow the same formula as before for this.


```python
%%writefile src/transform.py

import pandas as pd
from os.path import join
from extract import load_table
from load import save_data

def merge_truth_preds(df1, df2, **kwargs):
    return pd.merge(left=df1, right=df2, **kwargs)

if __name__ == "__main__":
    df1 = load_table(join("data", "interim"), "actuals_table.parquet")
    df2 = load_table(join("data", "interim"), "predicted_table.parquet")
    df_combined = merge_truth_preds(df1, df2, left_on="item_id", right_on="item_id", suffixes=("_truth", "_pred"))
    save_data(df_combined, join("data", "processed"), "combined_table.parquet")
```

Let's test our script to make sure everything is working properly.


```python
!python src/transform.py
```


```python
df_merged = pd.read_parquet(join("data", "processed", "combined_table.parquet"))
df_merged.tail()
```

### 3.3 Load

Note that more often than not data will be saved into a data warehouse or a data lake so that everyone in the team can access the files. With this in mind, let's update our load folder and mimic a data warehouse using DuckDB and Ibis. The former is a super fast in-memory database, and the latter is synthactic sugar for communicating with different databases.

Our `create_db` function will create a DuckDB database and a table to store our predictions.


```python
%%writefile src/load.py

from pathlib import Path
import pandas as pd
import ibis
from os.path import join

def create_db(path_in, path_out, file_name, table_name):
    path = Path(path_out)
    conn = ibis.duckdb.connect(path.joinpath(file_name))
    conn.register(path_in, table_name=table_name)
    print(f"Successfully loaded the {table_name} table!")

def save_data(data, path_out, file_name):
    path_out = Path(path_out)
    if not path_out.exists(): path_out.mkdir(parents=True)
    data.to_parquet(path_out.joinpath(file_name))
    print(f"Successfully loaded the {file_name} table!")
    
if __name__ == "__main__":
    create_db(
        path_in=join("data", "processed", "combined_table.parquet"),
        path_out=join("data", "dwarehouse"),
        file_name="db_analytics.ddb",
        table_name="truth_preds_challenge"
    )
```

Let's test it to make sure it works well.


```python
!python src/load.py
```


```python
import ibis
con = ibis.duckdb.connect(join("data", "dwarehouse", "db_analytics.ddb"))  # in-memory database
con.list_tables()
```


```python
data_preds = con.table("truth_preds_challenge")
data_preds.columns
```


```python
data_preds.to_pandas().head()
```

Now that we have processed the data, let's get started answering the questions for the challenge.

## 4. Evaluation Analysis

### 4.1 Confusion Matrix

The fastest way to get started analyzing the results coming out of a classification model is via a confusion matrix (CM). CMs provide us with a $2x2$ (or bigger) matrix where the columns are represented by the actual labels, and the rows by the predictions. Each element in the table represents the intersection of the two.

Here's a better way to visualize it. Imagine we are examining the predictions of a classification model sports cars.

|       | Actual Sports Car | Not a Sports Car|
|---|---|---|
| Predicted a Sports Car  | 420 |  80 |
| Predicted Not a Sports Car | 64 |  100 |

The way we evaluate these regions is as follows.
- `Actual Sports Car` and `Predicted a Sports Car` are considered `True Positives` since our model did a good jobs at predicting that a Ferrari is a sports car.
- `Actual Sports Car` and `Predicted Not a Sports Car` are considered `False Negatives` since our model mistakenly said that a Ferrari was the same as a Honda CRV, not a sports car.
- `Not a Sports Car` and `Predicted a Sports Car` are considered `False Positives` since our model predicted that Nissan Pathfinder was like a ferrari, a sports car.
- `Not a Sports Car` and `Predicted Not a Sports Car` are considered `True Negatives` since our model did a good jobs at predicting that Honda CRVs are not sports cars.

With our knowledge of confusion matrices, let's examine our predictions.


```python
mtx = m.confusion_matrix(df_merged.class_truth, df_merged.class_pred)
mtx
```


```python
conf = m.ConfusionMatrixDisplay(mtx)
conf.plot(colorbar=False);
```

As you can see from the image above, our model got
- 688 True Positives
- 0 True Negatives
- 45 False Negatives
- 19 False Positives

This means that, while our model did well detecting images with spores carryng a decease, it still got confused with a few images. This could potentially result in having a farmer spray some unnecessary pesticide in its farm or doing nothing at all while it should be taking action.

Let's examine a few other measures that help us understand the performance of our model.

### 4.2 Precision

**What is Precision?**
> Precision measures the percentage of samples that are correctly identified as positive out of all samples that the model identified as positive. In other words, it measures the proportion of true positives among all the samples that the model classified as positive.

$precision = \frac{TP}{TP + FP}$

Given the definition above, a high precision means that the model is good at avoiding false positives. Let's evaluate the precision of ours.


```python
round(
    m.precision_score(
        df_merged.class_truth, 
        df_merged.class_pred, 
        pos_label='Alternaria spp.'
    ) * 100,
    2
)
```

With a precision score of 93.86%, our models does very well at correctly classifying spores that carry out deceases while avoiding misclassifying those which don't have it.

### 4.3 Recall/Sensitivity

**What is Recall or Sencitivity?**
> Recall measures the percentage of all positive items that are correctly identified by the model. In other words, it measures the proportion of true positives that the model correctly identified among all the samples that are actually positive.

This means that if a model has a high recall it will be good at avoiding false negatives.

$recall = \frac{TP}{TP + FN}$

Let's evaluate the recall of our model.


```python
m.recall_score(df_merged.class_truth, df_merged.class_pred, pos_label='Alternaria spp.')
```

### 4.4 F1 Score

**What is an F1 Score?**

> The F1 score is a measure of a model's accuracy that combines the precision and recall metrics into a single score. It takes into account both false positives and false negatives, making it a useful metric for evaluating models that deal with imbalanced classes.

- $tp$ - True Positives
- $tn$ - True Negatives
- $fp$ - False Positives
- $fn$ - False Negatives

**F1 Formula**

$f1 = 2\frac{ (precision)(recall)}{precision + recall} = \frac{2 tp}{2tp + fp + fn}$

Let's get the F1 Score for our model.


```python
round(
    m.f1_score(
        df_merged.class_truth, 
        df_merged.class_pred, 
        pos_label='Alternaria spp.'
    ) * 100,
    2
)
```

As with precision and recall, the higher the score the better. In particular, a higher score means that our model can perform well on imbalanced data.

### 4.5 Accuracy

**What is Accuracy?**
> Accuracy is a metric used to evaluate how well a classification model is able to correctly predict the class label of unseen data points, and it is defined as the ratio of the number of correct predictions to the total number of predictions. 

While accuracy can be a very useful metric, it mostly shines with balanced datasets. Hence why we used more appropriate metrics such as precision, recall, or F1 score above.

Nonetheless, let's get the precision for our model.


```python
round(
    m.accuracy_score(
        df_merged.class_truth, 
        df_merged.class_pred
    ) * 100,
    2
)
```

### 4.6 Specificity

One last metric we could have calculated is Specificity.

**What is Specificity?**
> Specificity is a metric used to evaluate the ability of a classification model to correctly predict the negative class, and it is represented as the proportion of true negative predictions over the total number of actual negatives.

The above definition tells us that this metric is particularly useful in cases where the negative class is more important, such as in medical diagnosis, where correctly identifying healthy patients is critical. The problem with our analysis is that our **True Negatives** are 0, therefore, our specitivity will be $0$.

Similar to accuracy, specificity may not be a sufficient metric in cases of class imbalance, but it is still useful for understanding the performance of our model.

Even though we know it is $0$, let's use the formula to calculate it.

$specificity = \frac{tn}{tn + fp}$


```python
mtx[1][1] / (mtx[1][1] + mtx[0][1])
```

### 4.7 Autoevaluation

Let's start by creating functions for our metrics and finalize this section by turning these into a script as we have done before.


```python
def get_metrics(df, y_truth, y_pred, label):
    return dict(
        precision=round(m.precision_score(df[y_truth], df[y_pred], pos_label=label) * 100, 2),
        recall=round(m.recall_score(df[y_truth], df[y_pred], pos_label=label) * 100, 2),
        f1_score=round(m.f1_score(df[y_truth], df[y_pred], pos_label=label) * 100, 2),
        accuracy=round(m.accuracy_score(df[y_truth], df[y_pred]) * 100, 2)
    )
```


```python
metrics = get_metrics(df_merged, "class_truth", "class_pred", 'Alternaria spp.')
metrics
```


```python
def confused_mtx(df, y_truth, y_pred):
    return m.confusion_matrix(df[y_truth], df[y_pred])
```


```python
conf_mtx = confused_mtx(df_merged, "class_truth", "class_pred")
conf_mtx
```


```python
def generate_config(metrics, mtx, path=None, file_name=None):
    conf = OmegaConf.create({
        "facts": {
            "sample_size": int(sum(sum(mtx))),
            "healthy_preds": int(mtx[0][0]),
            "wrong_preds": int(mtx[1][0] - mtx[0][1])
        },
        "metrics": {k: float(v) for k, v in metrics.items()},
        "matrix": {
            "true_pos": int(mtx[0][0]),
            "true_neg": int(mtx[1][1]),
            "false_pos": int(mtx[0][1]),
            "false_neg": int(mtx[1][0])
        }
    })
    
    print(OmegaConf.to_yaml(conf))
```


```python
generate_config(metrics, conf_mtx)
```

We can now put everything together and finalize our last script.


```python
%%writefile src/evaluate.py

import sklearn.metrics as m
import pandas as pd
from os.path import join
from omegaconf import OmegaConf
from extract import load_table

def get_metrics(df, y_truth, y_pred, label):
    return dict(
        precision=round(m.precision_score(df[y_truth], df[y_pred], pos_label=label) * 100, 2),
        recall=round(m.recall_score(df[y_truth], df[y_pred], pos_label=label) * 100, 2),
        f1_score=round(m.f1_score(df[y_truth], df[y_pred], pos_label=label) * 100, 2),
        accuracy=round(m.accuracy_score(df[y_truth], df[y_pred]) * 100, 2)
    )

def confused_mtx(df, y_truth, y_pred):
    return m.confusion_matrix(df[y_truth], df[y_pred])

def generate_config(metrics, mtx, path, file_name):
    conf = OmegaConf.create({
        "facts": {
            "sample_size": int(sum(sum(mtx))),
            "healthy_preds": int(mtx[0][0]),
            "wrong_preds": int(mtx[1][0] - mtx[0][1])
        },
        "metrics": {k: float(v) for k, v in metrics.items()},
        "matrix": {
            "true_pos": int(mtx[0][0]),
            "true_neg": int(mtx[1][1]),
            "false_pos": int(mtx[0][1]),
            "false_neg": int(mtx[1][0])
        }
    })
    
    OmegaConf.save(conf, join(path, file_name))
    
    print(f"Config Successfully saved as {join(path, file_name)}")

if __name__ == "__main__":
    df = load_table(join("data", "processed"), "combined_table.parquet")
    metrics = get_metrics(df, "class_truth", "class_pred", 'Alternaria spp.')
    conf_mtx = confused_mtx(df, "class_truth", "class_pred")
    generate_config(metrics, conf_mtx, join("src", "configs"), "config.yml")
```

Let's test it to make sure everything is working correctly.


```python
!python src/evaluate.py
```


```python
!cat src/configs/config.yml
```

## 5. Building a Pipeline

We will be using dvc to create a reproducible pipeline, version, and cache artefacts.

In order for us to create a pipeline via different stages, we need to have run the following commands first (no need to do this since you probably cloned this repo):
- `dvc init`
- `dvc remote add -d local_data_lake data`

Now, we are ready to create our pipeline using `dvc stage add`, which will create a `dvc.yml` file which will track the different stages of our pipeline. Using the command `dvc repro` will run our pipeline and create a file called, `dvc.lock`, which will take care of versioning each component of it.

If you want to run these files for the first time you can either remove them with 
```sh
rm dvc.yml dvc.lock
```
or you can add `--force` in each of the steps, for example,
```sh
dvc stage add --force --name extract
```
or you can change some of the parameters like
```sh
dvc stage add --force --name extract \
    --deps data/raw/ground_truth/ \
    --deps data/raw/predictions/ \
    --outs your/data/dir/and/file.parquet \
    --outs your/data/dir/and/file.parquet \
    python src/extract.py
```


```python
# !rm dvc.lock dvc.yaml
```


```bash
%%bash

dvc stage add --name extract \
    --deps data/raw/ground_truth/ \
    --deps data/raw/predictions/ \
    --outs data/interim/actuals_table.parquet \
    --outs data/interim/predicted_table.parquet \
    python src/extract.py
```

Let's make sure our pipeline works by calling `dvc repro`.


```python
!cat dvc.yaml
```


```python
!dvc repro
```


```python
# we don't need to add each part of the pipeline manually
!dvc config core.autostage true
```

Let's add the last 3 sections of our pipeline.


```bash
%%bash

dvc stage add --name transform \
    --deps data/interim/actuals_table.parquet \
    --deps data/interim/predicted_table.parquet \
    --outs data/processed/combined_table.parquet \
    python src/transform.py
```


```bash
%%bash

dvc stage add --name load \
    --deps data/processed/combined_table.parquet \
    --outs data/dwarehouse/db_analytics.ddb \
    python src/load.py
```


```bash
%%bash

dvc stage add --name evaluate \
    --deps data/processed/combined_table.parquet \
    python src/evaluate.py
```


```python
!dvc repro
```

Finally, if we want to have a look at the graph created by our pipeline, we can `dvc dag` for that.


```python
!dvc dag
```

            +---------+          
            | extract |          
            +---------+          
                 *               
                 *               
                 *               
          +-----------+          
          | transform |          
          +-----------+          
             *       *           
           **         **         
          *             *        
    +------+        +----------+ 
    | load |        | evaluate | 
    +------+        +----------+ 
    [0m

## 6. Tests

The fastest way to get started testing our code is with `ruff`, a blazingly fast Python linter. In addition, we could go back and add type annotations to our scripts so that we can test code correctness with `mypy`.

It is important to note that we will use them here for example purposes only, and not to be included in a rigorous CI/CD pipeline (that's for another day 😎).


```python
!ruff check .
```

As you can see, `ruff` will alert us of any unused modules, long titles, and non-pythonic code found in our codebase, which becomes very useful when collaborating with others on large projects.


```python
!mypy src/extract.py
```

Because we have not created a local dev environment for our package, mypy will let us know that the load module is an unknown one for it and, therefore, we need to add such implementation as stub package to PyPI (similar to `pandas-stubs`).

There is also a type we need to fix, `List`, inside our `extract.py` scrip. We need to add `str` inside of it as `List[str]` so that mypy doesn't yell at us.

That was a quik intro to a few quick and dirty tests we can run on our codebase, there are plenty more we could take advantage of, of course.

## 7. Conclusion

We can take advantage of many open source tools to simulate the way in which we read, transform, load, and evaluate the output of our machine learning models. In addition, we can automate and version these pipelines using dvc while saving metrics in a generic config file for ease of use.

There are many approaches to model evaluation, and my hope is that you found this one useful.
