# Crops Challenge

You can view this project inside a deployed streamlit app or run it on binder using the links below. Conversely, you can run it on your local machine by following the setup section below.

-  [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bio-challenge.onrender.com/)
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ramonpzg/agri_challenge/HEAD?labpath=exploration.ipynb)
- For local development, please follow the instructions below.


## Set Up

### Conda Users

#### First Step

Open up your terminal and navigate to a directory of your choosing in your computer. Once there, run the following command to get the code for the session.

```sh
 git clone git@github.com:ramonpzg/agri_challenge.git
```

Conversely, you can click on the green `download` button at the top and donwload all files to your desired folder/directory. Once you download it, unzip it and move on to the second step.

#### Second Step

To get all dependencies, packages and everything else that would be useful to reproduce this project, you can recreate the environment by first going into the directory for the project.

```sh
cd agri_challenge
```

Then you will need to create an environment with all of the dependancies needed for the session by running the following command.

```sh
conda create -n bioscout python=3.10
conda activate bioscout
conda install --yes --file requirements.txt
# OR
pip install -f requirements.txt


## Conversely

python -m venv venv
source venv/bin/activate
pip install -f requirements.txt
```

#### Third Step

Open up Jupyter Lab and you should be ready to go.

```sh
jupyter lab
```