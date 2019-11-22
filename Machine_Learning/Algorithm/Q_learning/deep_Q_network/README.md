### Sepcs
* environment: python 3.6
* ide: jupyter notebook
* code: there are two .ipynb files in this folder: Train.ipynb and Test.ipynb. Train.ipynb trains the model and produces plots used in the report. Due to the nature of Q learning, it takes long time to learn the Q function. This file will take hours to run in my desktop with 32 G Memory and Intel i7-6700 4.00 GhZ (8 cores) processor. The plots presented in the report are also presented in this file for review. Test.ipynb load a model ("DQN_Lunar_Lander.h5") which was trained by Train.ipynb and then run tests on it. It is fast to run this file, only takes less than a minute. You can run 100 episodes of testing to get the testing curve or watch how the trained agent behaves for 10 episodes in the cartoon.
* model: "DQN_Lunar_Lander.h5" can be loaded by keras

### Requirements
* numpy==1.16.4
* pandas==0.24.2
* matplotlib==2.2.2
* gym==0.12.5
* Keras==2.2.4
* Keras-Applications==1.0.8
* Keras-Preprocessing==1.1.0
* tensorflow==1.4.0

### Run
1. open a command window and navigate to this folder
2. type "jupyter notebook" and hit enter, a local webpage will be pop up

#### Run Train.ipynb (not recommended unless sufficient computation power is available)
3. in the local webpage, double click Train.ipynb to open the file
4. read the title doc string in each code block and press shift + enter in each block as needed to run the code to train models and produce graphs, shift + enter is not needed if only review code and plots

#### Run Test.ipynb
3. in the local webpage, double click Test.ipynb to open the file
4. shift + enter on code block 1+2+3 to run 100 episodes testing to get the testing curve or shift + enter on code block 1+2+4 to watch how the trained agent behaves for 10 episodes, execution the code is not needed if only review plotss
