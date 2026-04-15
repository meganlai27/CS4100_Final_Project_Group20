# Music Note Generator
Members: Melvin Cheng, Megan Lai, Harsh Nigam

## Project Overview
We built a sheet music generator that will classify the notes and rhythms of a spectogram given an audio (.wav) file. Our product will then generate the MIDI file allowing the user to read and play along with the sound they heard.  

Some use cases for this would be 

By building a CNN and exploring the architectures, we were able to get a high accuracy to detect singular notes. 

## Existing Solutions


## Methodology

## Setup
```bash
pip install pretty_midi music21 librosa

# Install fluidsynth
choco install fluidsynth # for windows
```

## How to use:
`note_cnn.py` contains the definition for our `CNN` for a note classifier. 

To train the models, run the following:
```bash
python3 note_cnn.py # train the note classifier and save the model
python3 rhythm_cnn.py # train the rhythm classifier and save the model
```

By default, the trained models gets saved to `note_classifier.pth` and `rhythm_classifier.pth`.

## Results & Evaluation

### Note Classifier:
On training the note classifier, we get the following training loss over 10 epochs:
![loss](note_cnn_training_loss.png)

As we can see in the graph, the loss is steep in the first epoch, therefore the model learns the patterns in the spectograms quickly. 

To ensure the model is not overfitting the data, we also validate the data during training. As shown in the graph above, the validation loss is low throughout training epochs. The validation accuracy is depicted below. We can then see that the accuracy reaches 0.999 in the final training epoch.

![validation accuracies](note_cnn_validation_accuracies.png)

To further evaluate this on the test dataset, we can create a confusion matrix of the accuracy per note class.
![note classifier confusion matrix](./evaluation_results/note_cm.png)



## Resources

### Spectogram
https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53

### PyTorch
https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
