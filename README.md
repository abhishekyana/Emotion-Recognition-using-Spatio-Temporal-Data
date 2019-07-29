# Emotion-Recognition-using-Spatio-Temporal-Data
Emotion Recognition using Spatio-Temporal Data, applied on REVDESS Dataset, to predict 8 emotion, based on the Video(Spatio) and Audio(Temporal) Data.
* It is a multi-modal learning:
  * Where for the Spatial features a Convolutional Neural Networks are used.
  * For the Audio Temporal Data a variant of RNN called LSTM is used.
### Preprocessing the Spatio-Temporal data before feeding it to the model.
* For Spatial data: 
  * Video feed of 30 frames per second: We have 30 images per second. 
  * Uniformly sampled 5 images per second at regular intervals, like this.<br/>
  ![Images1](./images/ERimage.png)
  * So, Each image has a lot of white and unused space. So, as we need only the facial features of the person, I've applied Face Recognition to get the localised coordinated of the face and cropped the image to have only the face.<br/>
  ![Image2](./images/ERface.png)
  * All the five images are aligned horizontally to make a strip of images.
