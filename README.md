# Training Chatbot using Reinforcement Learning and LSTM
Using Reinforcement Learning and Recurrent Neural Networks to build a chat bot
The chatbot.
This is an End-toEnd solution we will create, train, test, deploy the model on the AWS infrastructure. 
I have also created a front end to communicate with the model through the rest api.

Here's an screen shot of the application:

![image](https://user-images.githubusercontent.com/32692718/79906435-fe3df480-83d4-11ea-9ad2-070472cb676c.png)
 
You interact with the application live from here.
The front endcode can also be accessed from here.
## Requirement
* python 3.5+
* pytorch 0.4.0


## Get started
#### Clone the repository
```
git clone https://github.com/khordoo/Reinforcement-Learning-LSTM-chartbot-pytorch
```
#### Corpus
I used the Cornell Movie-Dialogs Corpus dataset which contains structured dialogues extracted from various movie sources, and is known
.You can download it from here: https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.htm.
The dialogs in raw format does not provide a question-response format which we need to our chatbot. as a result,
 a wrote a custom script to perform some pre-processing to extract,parse and transform the data into a form suitable for training.
 Since the data set lso contains some metadata about the genre of the movies, I wrote a custom filtering option to be able to 
 extract the dialogs based on a specific genre.
 
The data needs to be downloaded and extracted into the data folder.Due to the relatively small amount of dataset, 
We keep the loaded dialogs in memory for fast processing and pass the processed dialog pait directly to our training routing for training. 


In here is a sample dialog from a comedy genre :
```
I'll see you next time.
Sure. Bye.
How are you?
Better than ever.
```

### References
1. [Recurrent Models of Visual Attention](https://arxiv.org/abs/1406.6247) by Volodymyr Mnih and others, 2014 
2. [Self-critical Sequence Training for Image Captioning](https://arxiv.org/abs/1612.00563) S. Rennie, Marcherett, and others in 2016