import streamlit as st
from PIL import Image

st.title("BioScout Technical Challenge")

st.sidebar.markdown("# Main page 📜")
st.sidebar.markdown("This page contains the challenge instructions. Please scroll through the sections above to see the answers")

st.markdown("""
## Machine Learning Engineer (take home):

### (1/2) Discussion Question:

At BioScout, high magnification microscope images are taken by our samplers of the air in farms 
every day. This data is uploaded to our cloud-side server where our labelling team labels the 
spores in each image. As BioScout scales, it becomes essential for machine learning to take 
over the core work of labelling.

**Task:**

In this question, we would like you to provide a discussion paragraph detailing how you would go 
about setting up a machine learning experiment to identify a particular spore type. Assume that a 
labelled dataset has been provided for you, including microscope images and labels of bounding 
boxes. Please outline the core tasks that would need to be completed in order to develop a machine 
learning model for this dataset.

**Expectations:**

The focus of this question is for you to outline the work that you would undertake to train a
strong model for the given dataset. Please outline the infrastructure or tools that you would
apply to achieve this to a production standard. Please also outline risks and challenges that
might be involved in the process.

In your response, please feel free to leverage any formats & communication tools that you
feel would be useful! We do not expect any code to be written for this question.

**Goal:**

In this question, we would hope for you to demonstrate your knowledge of ML processes,
tools and best practices.

""")

st.markdown("### (2/2) Programming Question:")

image1 = Image.open('images/raw_input.jpg')
image2 = Image.open('images/example_predictions.png')

col1, col2 = st.columns(2)

col1.subheader("Raw Image Sample")
col1.image(image1)

col2.subheader("Predicted Sample")
col2.image(image2)

st.markdown("""
**Task:**

The labellers have reviewed the output of a model, and now we would like to compare its
performance against the ground truth. Attached are the results of the model run on 752
images, as well as the labels generated by the labellers. Each file lists the predictions of a
single microscope image, with the filename denoting the image id. We would now like to
compare the difference between the two datasets. Please write an efficient code base to
output the F1 Score, Accuracy, Precision and Confusion Matrix of the dataset.

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

**Submission:**

There are two ways to submit your code:
1. Upload your repo to Github. Add ‘BioScout-Tom’ and ‘eltuna9’ as viewers to your repo.
2. Use git bundle to create a single file of your repo, and reply to this email with the bundle attached.

**Expectations:**
- When using a Jupyter notebook, please use markdown to clearly document each section.
- When using raw python, please make sure to commit frequently with clear commit messages, and to send through the bundled code.
- Bonus: Extra points for any other metrics that you believe would be useful for measuring the performance of the model.

**Goals:**

In this question, we hope for you to demonstrate your knowledge of Python applied to a
common ML data wrangling task, as well as modern software development practices.
""")