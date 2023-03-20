import streamlit as st
import pandas as pd
import graphviz
from omegaconf import OmegaConf
import seaborn as sns


st.title("Part 2 - Model Evaluation")
st.sidebar.markdown("# Programming Challenge")

cfg_main = OmegaConf.load('src/config.yml')
cfg_exp = OmegaConf.load('src/explanation.yml')


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

st.markdown("## Process for Creating Model and Predictions")
st.graphviz_chart(graph)


###########################
#         Facts           #
###########################

st.markdown("## Some Facts")

samp_size, healthy, pred_healthy = st.columns(3, gap="large")

samp_size.subheader("Sample Size")
samp_size.metric(label="Sample Size", value=752)
healthy.subheader("Healthy")
healthy.metric(label="Sample Size", value=752)
pred_healthy.subheader("Healthy Predictions")
pred_healthy.metric(label="Sample Size", value=752)


###########################
#    Metric #1 Accuracy   #
###########################

col1, col2 = st.columns([1, 4], gap="small")

col1.subheader(cfg_exp.explain.accuracy.title)
col1.metric(label=f"{cfg_exp.explain.accuracy.label}", value=cfg_main.metrics.accuracy)

col2.subheader(cfg_exp.explain.accuracy.subheader)
col2.markdown(cfg_exp.explain.accuracy.text)


###########################
#  Metric #2 Precission   #
###########################

col3, col4 = st.columns([1, 4], gap="small")

col3.subheader(cfg_exp.explain.precision.title)
col3.metric(label=f"{cfg_exp.explain.precision.label}", value=cfg_main.metrics.precision)

col4.subheader(cfg_exp.explain.precision.subheader)
col4.markdown(cfg_exp.explain.precision.text)


###########################
#    Metric #3 F1-Score   #
###########################

col5, col6 = st.columns([1, 4], gap="small")

col5.subheader(cfg_exp.explain.f1_score.title)
col5.metric(label=f"{cfg_exp.explain.f1_score.label}", value=cfg_main.metrics.f1_score)

col6.subheader(cfg_exp.explain.f1_score.subheader)
col6.markdown(cfg_exp.explain.f1_score.text)


###########################
#    Metric #4 Recall     #
###########################

col7, col8 = st.columns([1, 4], gap="small")

col7.subheader(cfg_exp.explain.recall.title)
col7.metric(label=f"{cfg_exp.explain.recall.label}", value=cfg_main.metrics.recall)

col8.subheader(cfg_exp.explain.recall.subheader)
col8.markdown(cfg_exp.explain.recall.text)


###########################
#  Metric #5 Specificity  #
###########################

col9, col10 = st.columns([1, 4], gap="small")

col9.subheader(cfg_exp.explain.specificity.title)
col9.metric(label=f"{cfg_exp.explain.specificity.label}", value=cfg_main.metrics.specificity)

col10.subheader(cfg_exp.explain.specificity.subheader)
col10.markdown(cfg_exp.explain.specificity.text)


###########################
#    Confusion Matrix     #
###########################

st.markdown("## Confusion Matrix")

mtx = pd.DataFrame([
    [cfg_main.matrix.true_pos, cfg_main.matrix.false_pos], 
    [cfg_main.matrix.false_neg, cfg_main.matrix.true_neg]
    ], columns=["Actual True", "Actual False"],
       index=["Predicted True", "Predicted False"]
)


fig = sns.heatmap(mtx, annot=True, fmt=".5g")
st.pyplot(fig.figure)