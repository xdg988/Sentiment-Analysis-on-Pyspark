from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes, SVMWithSGD
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.feature import Word2Vec
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.tree import DecisionTree
import csv

APP_NAME = "Sentiment"

if __name__ == "__main__":
    # Configuration
    conf = SparkConf().setAppName(APP_NAME)
    conf = conf.setMaster("yarn")
    sc = SparkContext(conf=conf)

    # File import
    originData = sc.textFile(u'hdfs:///dft/reviews.csv')     #We can change dataset here
    tipsData =  sc.textFile(u'hdfs:///dft/tips.csv')

    tipsDocument = tipsData.map(lambda line: line.split('\t')).filter(lambda line: len(line) >= 2).filter(lambda line: line[1]!='text')
    rateDocument = originData.map(lambda line: line.split('\t')).filter(lambda line: len(line) >= 2).filter(lambda line: line[0]!='stars')
    # Remember the number
    fiveRateDocument = rateDocument.filter(lambda line: int(float(line[0])) == 5).map(lambda line: (1, line[1]))
    fourRateDocument = rateDocument.filter(lambda line: int(float(line[0])) == 4).map(lambda line: (1, line[1]))
    threeRateDocument = rateDocument.filter(lambda line: int(float(line[0])) == 3).map(lambda line: (0, line[1]))
    twoRateDocument = rateDocument.filter(lambda line: int(float(line[0])) == 2).map(lambda line: (0, line[1]))
    oneRateDocument = rateDocument.filter(lambda line: int(float(line[0])) == 1).map(lambda line: (0, line[1]))
    allRateDocument = oneRateDocument.union(twoRateDocument).union(threeRateDocument).union(fourRateDocument).union(fiveRateDocument)

    # Generate training data
    rate = allRateDocument.map(lambda s: s[0])
    document = allRateDocument.map(lambda s: s[1].split(" "))
    tipsDocument = tipsDocument.map(lambda s: s[1])
    document_t = tipsDocument.map(lambda s: s.split(" "))

    hashingTF = HashingTF()
    tf=hashingTF.transform(document)
    tf.cache()

    idfModel = IDF().fit(tf)
    tfidf = idfModel.transform(tf)

    tf_t=hashingTF.transform(document_t)
    tf_t.cache()
    idfModel_t = IDF().fit(tf_t)
    tfidf_t = idfModel_t.transform(tf_t)
    training_t = tfidf_t


    zipped = rate.zip(tfidf)
    data = zipped.map(lambda line: LabeledPoint(line[0], line[1]))
    training, test = data.randomSplit([0.6, 0.4], seed=0)
#    LRmodel = LogisticRegressionWithSGD.train(training, iterations = 50)
    NBmodel = NaiveBayes.train(training, 1.0)
#    SVMmodel = SVMWithSGD.train(training, iterations=100)
    prediction = training_t.map(lambda p: (NBmodel.predict(p)))
    #We can get the accuracy here
#    predictionAndLabel = test.map(lambda p: (NBmodel.predict(p.features), p.label))
#    accuracy = 1.0 * predictionAndLabel.filter(lambda x: 1.0 if x[0] == x[1] else 0.0).count() / test.count()
#    print accuracy
    result = prediction.zip(tipsDocument)
    result1 = result.map(lambda line:(line[0], line[1]))
    result1.saveAsTextFile('tips_final_NB')
