import streamlit as st
import pandas as pd
import graphviz
from omegaconf import OmegaConf
import seaborn as sns


st.title("Part 2 - Model Evaluation")
st.sidebar.markdown("# Programming Challenge")

cfg_main = OmegaConf.load('src/configs/config.yml')
cfg_exp = OmegaConf.load('src/configs/explanation.yml')


###########################
#        Process          #
###########################

graph = graphviz.Digraph()
graph.attr(rankdir='LR', size='10,12')

graph.edge('Spores', 'Machine')
graph.edge('Machine', 'Data Lake')
graph.edge('Data Lake', 'Inference')
graph.edge('Data Lake', 'Split')
graph.edge('Split', 'Train')
graph.edge('Split', 'Validation')
graph.edge('Split', 'Test')
graph.edge('Train', 'Model')
graph.edge('Validation', 'Model')
graph.edge('Test', 'Evaluation')
graph.edge('Model', 'Evaluation')
graph.edge('Evaluation', 'Inference')

st.markdown("## 1. Process for Creating Model and Predictions")
st.graphviz_chart(graph)


###########################
#         Facts           #
###########################

st.markdown("## 2. Some Facts")
st.image("https://images.unsplash.com/photo-1466921583968-f07aa80c526e?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2670&q=80")

samp_size, healthy, pred_healthy = st.columns(3, gap="large")

samp_size.subheader("#️⃣ Sample Size")
samp_size.metric(label="-", value=cfg_main.facts.sample_size, label_visibility="collapsed")
healthy.subheader("💪🥗 Preds")
healthy.metric(label="-----------", value=cfg_main.facts.healthy_preds, label_visibility="collapsed")
pred_healthy.subheader("⛔ Predictions")
pred_healthy.metric(label="-----------", value=cfg_main.facts.wrong_preds, label_visibility="collapsed")


###########################
#        Metrics          #
###########################

st.markdown("## 3. Some Metrics")
st.image("https://images.pexels.com/photos/590041/pexels-photo-590041.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2")

###########################
#  Metric #1 Precission   #
###########################

col3, col4 = st.columns([1, 4], gap="small")

col3.subheader(cfg_exp.explain.precision.title)
col3.metric(label=f"{cfg_exp.explain.precision.label}", value=cfg_main.metrics.precision)

col4.subheader(cfg_exp.explain.precision.subheader)
col4.markdown(cfg_exp.explain.precision.text)


###########################
#    Metric #2 Recall     #
###########################

col7, col8 = st.columns([1, 4], gap="small")

col7.subheader(cfg_exp.explain.recall.title)
col7.metric(label=f"{cfg_exp.explain.recall.label}", value=cfg_main.metrics.recall)

col8.subheader(cfg_exp.explain.recall.subheader)
col8.markdown(cfg_exp.explain.recall.text)


###########################
#    Metric #3 F1-Score   #
###########################

col5, col6 = st.columns([1, 4], gap="small")

col5.subheader(cfg_exp.explain.f1_score.title)
col5.metric(label=f"{cfg_exp.explain.f1_score.label}", value=cfg_main.metrics.f1_score)

col6.subheader(cfg_exp.explain.f1_score.subheader)
col6.markdown(cfg_exp.explain.f1_score.text)


###########################
#    Metric #4 Accuracy   #
###########################

col1, col2 = st.columns([1, 4], gap="small")

col1.subheader(cfg_exp.explain.accuracy.title)
col1.metric(label=f"{cfg_exp.explain.accuracy.label}", value=cfg_main.metrics.accuracy)

col2.subheader(cfg_exp.explain.accuracy.subheader)
col2.markdown(cfg_exp.explain.accuracy.text)


###########################
#    Confusion Matrix     #
###########################

st.markdown("## 4. Confusion Matrix")
st.image("images/conf_mtx.png")

st.markdown("""
The fastest way to get started analyzing the results coming out of a classification model is via a confusion matrix (CM). CMs provide us with a $2x2$ (or bigger) matrix where the columns are represented by the actual labels, and the rows by the predictions. Each element in the table represents the intersection of the two.

Here's one way to understand it. Imagine we are examining the predictions of a classification model sports cars.

|       | Actual Sports Cars | Not Sports Cars|
|---|---|---|
| Predicted Sports Cars  | 420 |  80 |
| Predicted Not Sports Cars | 64 |  100 |

The way we evaluate these regions is as follows.
- `Actual Sports Car` and `Predicted a Sports Car` are considered `True Positives` since our model did a good jobs at predicting that a Ferrari is a sports car.
- `Actual Sports Car` and `Predicted Not a Sports Car` are considered `False Negatives` since our model mistakenly said that a Ferrari was the same as a Honda CRV, not a sports car.
- `Not a Sports Car` and `Predicted a Sports Car` are considered `False Positives` since our model predicted that Nissan Pathfinder was like a ferrari, a sports car.
- `Not a Sports Car` and `Predicted Not a Sports Car` are considered `True Negatives` since our model did a good jobs at predicting that Honda CRVs are not sports cars.

With our knowledge of confusion matrices, let's examine our predictions.
""")

mtx = pd.DataFrame([
    [cfg_main.matrix.true_pos,  cfg_main.matrix.false_pos], 
    [cfg_main.matrix.false_neg, cfg_main.matrix.true_neg]
], 
    columns = ["Actual True", "Actual False"],
    index   = ["Predicted True", "Predicted False"]
)


fig = sns.heatmap(mtx, annot=True, fmt=".5g")
st.pyplot(fig.figure)