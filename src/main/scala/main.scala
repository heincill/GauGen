import breeze.linalg.{DenseMatrix, DenseVector}

object main {
  def main(args: Array[String]): Unit = {
    val pso: MGPSO = MGPSO(30, 0.729844, 1.496180, 1.496180)

//    for (d <- List(2)) {
//      for (k <- List(2)) {
//        for (i <- List(10, 50, 100, 250, 500, 1000)) {
//          val res: Vector[Double] = (1 to 30).toVector.map(_ => pso.optimize(3, GaussianMixture.genCovarianceMatrices(d, Vector.fill(k)(1), Vector.fill(k)(1)), i)._1)
//          println(s"d: $d | k: $k | i: $i => m: ${mean(res)} | dev: ${dev(res)}")
//        }
//      }
//    }

    val K = 4
    val D = 2

    val w = GaussianMixture.genWeights(K, 1)
    val s = GaussianMixture.genCovarianceMatrices(K, Vector.fill(D)(1), Vector.fill(D)(1))
    val m = GaussianMixture.genMeanVectors(3, s)
    val data = GaussianMixture.genMixture(1000, w, m, s, false)
    Visualizer.clust(data, m, s, "REEEEE")
  }

  def mean(x: Vector[Double]): Double = x.sum / x.size

  def dev(x: Vector[Double]): Double = {
    val m: Double = mean(x)
    math.sqrt(x.map(e => math.pow(e - m, 2)).sum)
  }
}
