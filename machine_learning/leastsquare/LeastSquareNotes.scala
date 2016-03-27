package com.youku.data.algorithm.leastsquare

import breeze.linalg._

//Reference Documentation : http://blog.csdn.net/qll125596718/article/details/8248249
object LeastSquareNotes {
  var a:Double = 0.0
  var b:Double = 0.0

  def getLeastSquare(x:Vector[Double],y:Vector[Double]): (Double,Double) = {
    var t1 = 0.0
    var t2 = 0.0
    var t3 = 0.0
    var t4 = 0.0

    for (i <- 0 until x.size) {
      t1 += x(i) * x(i)
      t2 += x(i)
      t3 += x(i) * y(i)
      t4 += y(i)
    }
    a = (t3 * x.size - t2 * t4) / (t1 * x.size - t2 * t2)
    b = (t1 * t4 - t2 * t3) / (t1 * x.size - t2 * t2)
    (a,b)
  }
  def print(): Unit ={
      println("y = " + a + "x + " + b)
  }

  def getY(v:Double): Double ={
    a * v + b
  }

  def main(args: Array[String]) {
        val in1 = Array(1.0,2.0,3.0)
        val xx =  new DenseVector[Double](in1)
        val in2 = Array(5.0,9.0,10.0)
        val yy = new DenseVector[Double](in2)
//        val tup = getLeastSquare(xx,yy)
//        print(tup._1,tup._2)
        getLeastSquare(xx,yy)
        print
        println(getY(4.0))

    /**
      * y = 2.5x + 3.0
      *13.0
      */

  }
}
