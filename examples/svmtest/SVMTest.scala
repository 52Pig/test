package com.youku.data.algorithm.test

import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

object SVMTest {
//  lazy private val md5enc = java.security.MessageDigest.getInstance("MD5")
  def main(args: Array[String]) {
//    902c82d3f883e8ce5ffeaa74cb2bc3b9	0	31	9	1	0	0	11	3	2	1	09	0	0	0	0	0	85.83333333333333	41488326	1	1	3.7
    //    val conf = new SparkConf().setAppName("Classfy Test").setMaster("spark://a268.datanode.hadoop.qingdao.youku:7077")
    val conf = new SparkConf().setAppName("Classfy Test").setMaster("local")
    val sc = new SparkContext(conf)
//    val data = sc.textFile(args(0))
    val data = sc.textFile("E:/Items/anticheat/behavior/source_20160612.txt")

    val parsedData = data.map{ line =>
        val parts = line.split("\t").drop(1).init
        val a = Array(parts(0),parts(1),parts(2),parts(3),parts(4),parts(5),parts(6),parts(7),parts(8),parts(9),parts(11),parts(12),parts(13),parts(14),parts(15),parts(16),parts(17),parts(18),parts(19))
//        1	4	4	1	0	0	1	1	1	0	-1	1	0	0	0	0	1.3333333333333333	577410384	1	1	3.7.1.0
//        parts.mkString("\t")
//        LabeledPoint(parts(0).toInt , Vectors.dense(parts.tail.map(_.toDouble)))
        LabeledPoint(a(0).toInt , Vectors.dense(a.tail.map(_.toDouble)))
    }
//    parsedData.foreach(println)
//    exit()
//    (1.0,[1.0,5.0,4.0,1.0,0.0,0.0,2.0,1.0,1.0,0.0,-1.0,1.0,0.0,0.0,0.0,0.0,10.0,5.77410384E8,1.0,1.0])
    //设置迭代次数
    val numIterations = 20
    val splits = parsedData.randomSplit(Array(0.7,0.3),seed=11l)
    val training = splits(0).cache
    val test = splits(1)
//    println(training.count()+":::"+parsedData.count())  //308402:::440554
    val model = SVMWithSGD.train(training, numIterations)
//    println("截距: "+ model.intercept)
//    println("权重: "+ model.weights)
//    println("阀值: " + model.getThreshold)
//    println(model.toPMML())
  /**
    * 截距: 0.0
权重: [1.6610391684749493,-187.66036954970798,-18.938307846934617,-2.942445057492003,0.0,0.0,-24.585686168976817,-15.07723773311887,-15.841073768683108,-15.349049759356985,-19.724642799823872,1.1312208626604325,0.0,-2.376500940972157,-1.2788613090907066,-0.01791889786957251,-431.8420443591006,3.411001945841929E7,-1.475464630445227,-0.18369665782919992]
阀值: Some(0.0)
    */
//    val svmAlg = new SVMWithSGD()
//    svmAlg.optimizer.setNumIterations(200).setRegParam(0.1).setUpdater(new L1Updater)
//    val modelL1 = svmAlg.run(training)
//    val model = SVMWithSGD.train(training,numIterations,0.01,0.01)
//    model.clearThreshold()

//    println("weight: " + model.weights)
//    println("intercept: " + model.intercept)
//    println("pmml : " + model.toPMML)
    val labelAndPreds = test.map{point=>
        val prediction = model.predict(point.features)
        (point.label,prediction)
    }
    /*val trainErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble/parsedData.count
    println("Training Error : " + trainErr)   //Training Error : 0.09701194405226148

    val metrics = new BinaryClassificationMetrics(labelAndPreds)
    val auROC = metrics.areaUnderROC()
    println("Area under ROC " +auROC)  // Area under ROC 0.7173437159846103*/

    //P正元组数   N负元组数
    val P = labelAndPreds.filter(r=>r._1==1).count
    val N = labelAndPreds.filter(r=>r._1==0).count

    println("P:"+P)
    println("N:"+N)

    val TP = labelAndPreds.filter(r=>r._1==1&&r._2==1).count.toDouble  //被分类器正确分类的正元组
    val TN = labelAndPreds.filter(r=>r._1==0&&r._2==0).count.toDouble //被分类器正确分类的负元组
    val FP = labelAndPreds.filter(r=>r._1==0&&r._2==1).count.toDouble  //被错误地标记为正元组的负元组
    val FN = labelAndPreds.filter(r=>r._1==1&&r._2==0).count.toDouble //被错误地标记为负元组的正元组

    println("TP:" + TP)
    println("TN:" + TN)
    println("FP:" + FP)
    println("FN:" + FN)

    val accuracy = (TP + TN)/(P+N)
    val precision = TP/(TP+FP)
    val recall = TP/P

    //准确率
    println("accuracy:" + accuracy)
    //精度
    println("precision:" + precision)
    //召回率
    println("recall:" + recall)

    val F = (2 * precision * recall) / (precision + recall)
    println("F:"+F)

    /**TP:88775.0
    TN:638.0
    FP:42536.0
    FN:203.0
    accuracy:0.6765921060596889
    precision:0.6760667423140483
    recall:0.9977185371664906
    F:0.8059866811325124
    */
  }
}
