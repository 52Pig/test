package com.youku.data.algorithm.sensemachine

import breeze.linalg.{DenseMatrix, DenseVector}


object SenseMachine {

  def main(args: Array[String]) {
    sensemachine1
  }
//http://www.tuicool.com/articles/YRN3QzF
  def sensemachine1(): Unit ={
    val testX = DenseMatrix.zeros[Double](3,2)
    testX(0 to 2,0 to 1) := DenseMatrix((3.0,3.0),(4.0,3.0),(7.0,1.0))
//    testX(0 to 2) := DenseVector()
//    println(testX)
    val c = testX(::,0)

    val testY = DenseVector.zeros[Int](3)
//    println(testY(0 to 1) )
    testY(0 to 2) := DenseVector(1,1,-1)
//    println(testY)
//    Utils.cross_product(c,testY)
    Perception.train(testX,testY,3)
    Perception.printModel()


  }
}

object Utils {

  def cross_product(a: DenseVector[Double], b: DenseVector[Double]): Double = {
    var result = 0.0
    if (a.length == b.length) {
      for (i <- 0 until a.length) {
        result += a(i) * b(i)
      }
    }
    result
  }
}
class Perception private (
     private var iters:Int,
     private var learnRate:Int,
     private var initw:Double,
     private var initb:Double) {
  def this() = this(100,1,0.01,0.0)
//  override def toString = "iters: "+iteration + " learnRate: "+learnR + "initw: " + initW +" initb: " + initB
//  def this(iters:Int,learnRate:Int,initw:Double,initb:Double) = { this(iters,learnRate,initw,initb) }
}
object Perception {

  var w = DenseVector.zeros[Double](3)
  var step = 0.01
  var b = 0.0

  def apply(iters:Int,learnRate:Int,initw:Double,initb:Double) = new Perception(iters,learnRate,initw,initb)

  def train(trainX:DenseMatrix[Double],trainY:DenseVector[Int],iterations:Int): Boolean ={
      require(trainX.rows == trainY.length)

      for(iter <- 1 to iterations){
        var flag = true
        for(i <- 0 until trainX(::,0).length){
//          if(trainY(i) * (Utils.cross_product(w,trainX(::,i))+b) <= 0 ){
//          val c = trainX(::,i)
          if((trainY(i) * Utils.cross_product(w,trainX(::,0))) <= 0){
            updateWB(trainX(::,0),trainY(i))
            flag = false
          }
        }
        if(flag) true
      }
    false
  }

  def printModel(): Unit ={
    for(m  <- 0 to w.length-1 ){
      println(w(m) + "x" + b)
    }
  }

  def updateWB(w:DenseVector[Double],y:Double): Unit ={
    for(i <- 0 until w.length){
      w(i) += step * y * w(i)
    }
    b += step * y
  }
}