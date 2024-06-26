# Classification-with-Single-Layer-Neural-Networks

Part A: Discrete-neuron Perceptron

For this part of homework, you will train and test a single-layer discrete-neuron perceptron neural network to classify the English letters. In this dataset, there are 20 samples (with different fonts) of English alphabet letters, stored in 26 folders. Each letter is a PNG image file of size 60×60.

1. Convert every PNG image into bipolar values of +1 for white and -1 for non-white (black) pixels.
2.  For this part (discrete-neuron perceptron), assume each pixel of letter as a bipolar feature (n=3600), and use them as input to Neural Network.
3. Train your Neural Network model using the training data and then classify the test letters.
4. To examine the generalization ability, use leave-one-out cross validation (LOOCV) method and report the accuracy. In LOOCV on N data samples, each time one sample is set aside for testing the model while all remained samples are used for training it. This procedure is repeated N times until all samples are chosen as test once. The average of accuracies will be the accuracy of model.
5. To examine its robustness to noise, train the network using all training letters and then test it using degraded training letters with 15% and 25% of noise (only toggle the black pixels of the letters) as new test data and report the accuracy.

   
Part B: Continuous-neuron Perceptron

For this part, you will train and test a single-layer continuous-neuron perceptron neural network to classify the English letters. This time, assume each letter is a 60×60 grayscale image with 256 intensities (0 to 255) for each pixel and so, use continuous-neuron perceptrons.
