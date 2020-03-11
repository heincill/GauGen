import breeze.linalg.{DenseMatrix, DenseVector, diag, eigSym}
import breeze.plot._

object Visualizer {

  def graph(data: Vector[DenseVector[Double]]): Unit = {
    val X: DenseMatrix[Double] = new DenseMatrix(data.head.length, data.size, data.foldLeft[Array[Double]](Array())(_ ++ _.data))
    val f = Figure()
    val p = f.subplot(0)
    p += plot(X(0,::).inner, X(1,::).inner, '.', colorcode="black")
    f.saveas("data.png")
  }

  def clust(data: Vector[DenseVector[Double]], mu: Vector[DenseVector[Double]], sigma: Vector[DenseMatrix[Double]], alg: String): Unit = {
    val X: DenseMatrix[Double] = new DenseMatrix(data.head.length, data.size, data.foldLeft[Array[Double]](Array())(_ ++ _.data))
    val f = Figure()
    var p = f.subplot(0)
    p.title = "Algorithm " + alg
    p += plot(X(0,::).inner, X(1,::).inner, '.', colorcode="black")

    val alpha_vals: List[Double] = List(0.68, 0.95, 0.05)
    for (i <- alpha_vals; k <- mu.indices) {
      val s = -2 * math.log(1-i)
      val eigen_info = eigSym(sigma(k) * s)
      val t = breeze.linalg.linspace(0, 2 * math.Pi)
      val T: DenseMatrix[Double] = new DenseMatrix[Double](2, t.length, t.data.flatMap(e => Array(math.cos(e), Math.sin(e))))
      val a: DenseMatrix[Double] = eigen_info.eigenvectors * diag(new DenseVector(eigen_info.eigenvalues.data.map(e => math.sqrt(e)))) * T
      p += plot(a(0,::).inner + mu(k)(0), a(1,::).inner + mu(k)(1), colorcode="red")
    }
    f.saveas("clusters.png")
  }

  def Plot_Cluster_Measure(measure: String, x: Vector[Double], y: Vector[Double]): Unit = {
    val f = Figure()
    var p = f.subplot(0)
    p.title = measure
    p += plot(x, y, colorcode="black")
    //p += plot(X(0,::).inner, X(1,::).inner, colorcode="black")
    f.saveas("clusters.png")
  }
}