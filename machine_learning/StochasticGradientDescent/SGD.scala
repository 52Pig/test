object SGD{
  val data = HashMap[Int,Int]()
  def getData():HashMap[Int,Int] = {
    for(i <- 1 to 50) {
	  data += (i -> (12 * i))
	}
    data
  }
  var theta:Double = 0
  var a:Double = 0.1
  
  def sgd(x:Double,y:Double){
	theta = theta - alpha * ((theta * x) - y)
  }
  
  def main(args: Array[String]) {
    val dataSource = getData()
	dataSource.foreach(myMap => {sgd(myMap._1,myMap._2)})
    println("最终结果theta值为：" + theta)
  }
}