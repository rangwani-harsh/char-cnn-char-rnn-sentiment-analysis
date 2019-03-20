## Character Level Models For Sentiment Analysis
The CNN version is the same as Yoon Kim's CNN applied at character level. The char RNN is a GRU based model. 
Properties:

1) No Preprocessing 
2) Two models (One is char cnn and the other is char rnn).
3) Evaluation metric - Macro F1
4) Saving the checkpoints if validation_macro_f1 > best_macro_f1.


## To evaluate the validation accuracy:


```python main.py --test --snapshot saved-models/best-cnn.pt```  
```python main.py --test -snapshot saved-models/best-rnn.pt --rnn```  

## For prediction of the sentence file:

```python predict.py --input input_file -output output_file```

Suggestions for improving the model
1) Hyperparameter tuning
2) Preprocessing
3) Char CNN and Char LSTM can be used on token level as being used as a whole.
4) Explore class_weight parameter to make our classifier work equally well on all classes.

```
For CNN best validation F1 is around 73%. 
              precision    recall  f1-score   support

           0     0.8567    0.8858    0.8710       911
           1     0.5943    0.6176    0.6058       306
           2     0.7990    0.6599    0.7228       247

   micro avg     0.7917    0.7917    0.7917      1464
   macro avg     0.7500    0.7211    0.7332      1464
weighted avg     0.7921    0.7917    0.7906      1464
```

```
For Char RNN validation F1 is around 70%
              precision    recall  f1-score   support

           0     0.8448    0.8606    0.8526       911
           1     0.5569    0.5915    0.5737       306
           2     0.7251    0.6194    0.6681       247

   micro avg     0.7637    0.7637    0.7637      1464
   macro avg     0.7090    0.6905    0.6982      1464
weighted avg     0.7645    0.7637    0.7632      1464

0- negetive 1-neutral 2-positive
```

Training was done on Titan XP GPU. As the training and validation sets are being determined at run time I have tried to keep the seeds same and it works on my system however I haven't used deterministic version of cuda. 

The boiler plate code was taken from the repository https://github.com/srviest/char-cnn-text-classification-pytorch.

In case you require assistance please feel free to email me harsh.rangwani.cse15@iitbhu.ac.in

Install the requirements by pip install -r requirements.txt
