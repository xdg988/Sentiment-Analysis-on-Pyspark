
# Sentiment-Analysis-on-Pyspark

** I have to admit that it is hard to run our code in your machine. Because all this things are done on our university's powerful machines and all the configurations(like Hadoop and Spark) are deployed during the course. We just want to record what our group has done and successfully finished this hard project. For all detail information, please refer to our report PPT. **

This is a course project about cluster and cloud computing. The whole process trains a simple machine learning model for sentiment classification task on Spark.

Basically, The task is based on the `review.json` from  [Yelp Dataset](#https://www.yelp.com/dataset) file (about 6 million rows). We only need two fields in this file, namely "stars" and "text". The "text" field is customer's raw review sentence, while the "stars" field is the customer's rating for the corresponding review ranging from 1 to 5. We applied the model to a binary task. In the binary setting, reviews with star greater than 3 are regarded as positive samples, otherwise as negative ones.

The model architecture is described in our report. Since our group members are all not familier with Pyspark API, we only use very simple ways to train our model. Firstly, we use TF-IDF to train the texts and convert them into word vector. Then we use LR(with different iterations), SVM(with different iterations) and NB to train our classification models. Finally, we predict reviews from `tips.csv` in [Yelp Dataset](#https://www.yelp.com/dataset). The model achieved up to 89% accuracy on the validation set. For more details and results, please refer to our PPT.

Below is the readme-file that we submitted to our course professor.

# Statement

There may be two different kinds of clients who use our application.
* One of them may just want to do some predict of the text file to make sentiment analysis, what she/he can just change the name of “tip.csv” to the “xxx.csv” which stores her/his own text[Using our dataset to train model].
* Another kind of client may want to train the model by their own data(hereafter marked as client_train_data) and then make sentiment analysis based on their own text(client_analysis_data). For these clients, Client_train_data should contains two attributes: text and emotion(posit

# 1. Using our project’s data

Our project’s data is review.json and tips.json of Yelp.

## 1.1 Data Pre-processing and Uploading

* Setup YARN/HDFS/Spark environment using the given configurations.

* Download `review.json` in [Yelp Dataset](#https://www.yelp.com/dataset), or you can directly use `review.json` in our student71 machine.

* Run `json2bigcsv.py`(in python3 environment) to convert json file into csv file and extract features we need.

* Run `hdfs dfs -put reviews.csv /dft` to upload the csv file to HDFS.


## 1.2 Model training and Predict

* Input cd in the machine to enter the descktop.

* Model training and predict(sentiment analysis).

There are two choices here, for clients wanting higher accuracy or faster speed:<br />Higher Accuracy: `run /opt/spark-2.4.0-bin-hadoop2.7/bin/spark-submit --master=yarn ~/sentiment_SVM100.py`<br />Faster Speed: `run /opt/spark-2.4.0-bin-hadoop2.7/bin/spark-submit --master=yarn ~/sentiment_NB.py`<br />[PS: This result is just based on our own data(`review.json`), sometimes use other algorithms rather than SVM100 may get higher accuracy, while NB algorithm is always fastest according its calculation.]


## 1.3 Project’s website

Open Sentiment Analysis on Yelp.html in the “web” fold. There are four main functions about our web as following. We can visit the Spark monitor web interface by Spark link, visit the Ganglia web interface by Ganglia link, visit the Hadoop resource manager web interface by Hadoop link. If we want to download the predicted result, we can click the Output link.
<img src="1.png" width="40%">

# 2. For Clients in statement 1

* Preprocess her/his own data set and submit to HDFS and change the position of their data set in the corresponding code of `sentiment_NB.py` or `sentiment_SVM100.py`[Displayed as following]. For these clients, they just need to change the position behind the variable "tipsData."<br />
* Run one of these python file based on their own choice.<br />
* Monitor the process or result by our website, following the part in 1.3.

# 3. For Clients in statement 2

* Preprocess her/his own data set and submit to HDFS and change the position of their data set in the corresponding code of `sentiment_NB.py` or `sentiment_SVM100.py`[Displayed as following]. For these clients, they should find the position behind the variable “originData”[Sample Set] and "tipsData"[Test Set].<br />
* Run one of these python file based on their own choice.<br />
* Monitor the process or result by our website, following the part in 1.3.

