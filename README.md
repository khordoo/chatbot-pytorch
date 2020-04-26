# seq2seq Chatbot implemented in Pytorch
In this repo we are going to build a conversational chatbot using Recurrent Neural Networks. The recurrent networks uses the encoder-decoder architecture with attention.
This is an end-to-end solution in which we start with building the seq2Seq model in PyTorch, then train and evaluate the model using Cornell movies dialog dataset and finally deploy the trained model as an api endpoint using Nginx and Flask on AWS infrastructure. 
I also created a frontend web application to have a convenient user interface for communicating with our chatbot.

Here's an screen shot of the application:

![image](https://user-images.githubusercontent.com/32692718/80289273-d9aa8b00-86fa-11ea-80c4-68c806369edd.png)

 
You can interact with the chatbot live from here.

The frontend app was implemented using Vuetify.js and the code is also available on my [github](https://github.com/khordoo/chatbot-frontend)
## Requirement
* python 3.5+
* pytorch 0.4.0


## Get started
#### Clone the repository
```
git clone https://github.com/khordoo/chatbot-pytorch
```
#### Corpus
I used the Cornell Movie-Dialogs Corpus dataset which contains structured dialogues extracted from various movie sources, and is known
.You can download it from here: https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.htm.
The dialogs in raw format does not provide a question-response format which we need to our chatbot. as a result,
 a wrote a custom script to perform some pre-processing to extract,parse and transform the data into a form suitable for training.
 Since the dataset also contains some metadata about the genre of the movies, I wrote a custom filtering option to be able to 
 extract the dialogs based on a specific genre.However, the bot is trained on movie dialogs from all the genres.
 
The data needs to be downloaded and extracted into the data folder.
```
chatbot-pytorch/data
```

Due to the relatively small amount of dataset, 
We keep the loaded dialogs in memory for fast processing and pass the processed dialog directly to our training routing for training. 


In here is a sample dialog from a comedy genre :
```
Why are you doing this?
I just wanted to keep an eye on you. 
Where did everything go?'
I sold it all at auction.
Seriously?
Yes.
```

To make the model converge faster, some filtering were applied to the data. 
- Relatively shorter sentences were used for training. ( Max length of 20) 
- Vocabulary size was reduced

During the data loading, the we count of occurrences for every word
in the dictionary, if the occurrence is fewer than 3 , they are removed from the dictionary. Subsequently 
if a sentence contains any removed word it will not be included in the training data.

### BLEU Score
To keep track of the training progress, in addition to loss, the BLEU score is calculated during the training. For this the bleu_score function
from the NLTK library is used.

### Training
We split the data into the train and test dataset and keep track of the BLEU score and loss during the training. 

### Tracking the training
In addition to the logs that are written to the screen, you can also track the progress of the training in the Tensorboard.
To view the loggs in Tensorboard run the following commands:

````shell script
tensorboard --logdir runs/
````
In case you decided to run the code in a notebook .Run the fallowing command before starting the training:
```shell script
%load_ext tensorboard
%tensorboard --logdir runs/
````

The following metrics are written to the Tensorboard:
- Mean training loss
- Mean Training Bleu score
- Mean testing Bleu score

Here is the progress of the bleu score over time for the first few epochs.
![image](https://user-images.githubusercontent.com/32692718/80285109-deae1100-86df-11ea-8d85-d428a6d71cd3.png)

##Loading the saved model for inference
The checkpoints are being saved to disk during the training. The saving frequency 
is being controlled by the configuration variable *SAVE_CHECKPOINT_EVERY*

BY default the files are being saved in **saves** directory. this location can be changed using *SAVE_DIR* variable.

The saved file can be specified in the predict.py model for inference.



### References
1. [Recurrent Models of Visual Attention](https://arxiv.org/abs/1406.6247) by Volodymyr Mnih and others, 2014 
2. [Self-critical Sequence Training for Image Captioning](https://arxiv.org/abs/1612.00563) S. Rennie, Marcherett, and others in 2016