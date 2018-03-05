The folder contains:
1. RNN.py: Main code
2. RNN.ipynb: Helps view the training process better
3. download.py: Script to download model weights
4. makefile: Creates required folders and runs download.py
5. ReadMe.txt: Instructions to use the code + Report

Steps to build:
1. Run the 'make' command to download the model weights. 
2. Place the data in a folder named 'data'.
Note: data folder should be present in the parent directory (..) of the Assignment
directory (.).
3. Run the code.

To train, enter command: 'python2 RNN.py --train --model y --hidden_unit n'
	* y can be lstm or gru
	* the command trains the corresponding model with required hidden unit size
	e.g.: python2 RNN.py --train --model lstm --hidden_unit 100

Note: If you have way too much time on your hands, and really want to train the model
using our code, we suggest that you train with learning rate 1e-3, 1e-4 and then 5e-5 
for 10, 15 and 20 epochs respectively was the best way to go with it. So, that's what 
the train mode does. 

To test, enter command: 'python2 RNN.py --test --model y --hidden_unit n'
	* y can, again, only be lstm or gru
	* it reloads the model weights from the relevant logbook and evaluates model
	e.g. python2 RNN.py --test --model lstm --hidden_unit 100

We've already trained the model for y = lstm, gru, n = 50, 100, 200, and achieved the following results:

Hidden layer size vs Test accuracy
LSTM:
	50: 83.38%
	100: 84.59%
	200: 85.95%

GRU:
	50: 81.00%
	100: 82.76%
	200: 84.57%
