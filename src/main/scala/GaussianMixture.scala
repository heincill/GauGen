import spire.math
import breeze.linalg.{DenseMatrix, DenseVector, diag, inv, qr}
import breeze.stats.distributions.{Bernoulli, MultivariateGaussian, Uniform}
import scala.util.Random


/**
  * Style change will be slow to implement due to unforseen circumstances.
  */
object GaussianMixture {

  def genMixture(
                  N: Int,
                  w: Vector[Double],
                  m: Vector[DenseVector[Double]],
                  s: Vector[DenseMatrix[Double]],
                  O: Boolean): Vector[DenseVector[Double]] = {
    val K: Int = w.size

    // Divide
    val clust_samples: Vector[Int] = w.map(e => Math.round(e * N).toInt)

    // Generate list of gaussian data
    val pre_data: Vector[Vector[DenseVector[Double]]] = (0 until K).map(k => Vector.fill(clust_samples(k))(MultivariateGaussian(m(k), s(k)).draw())).toVector

    // Flatten gaussian data
    var data: Vector[DenseVector[Double]] = pre_data.flatten

    // Add outliers
    if (O) data = Add_outliers(pre_data, m, s)
    data
  }

  // Generate a list of mixture coefficients
  def genWeights(K: Int, r: Double): Vector[Double] = {
    if (K < 2) throw new Exception("Invalid number of components.")
    if (r < 1) throw new Exception("Mixture coefficient eccentricity cannot be less than one.")
    val n: Vector[Double] = Vector(1, r) ++ Vector.fill(K - 2)(Uniform(1, r).draw())
    val w: Vector[Double] = n.map(x => x / n.sum)
    w
  }

  // Generate covariance matrices
  def genCovarianceMatrices(K: Int, ecc: Vector[Double], maxDev: Vector[Double]): Vector[DenseMatrix[Double]] = {
    if (ecc.size != maxDev.size) throw new Exception("Unequal eccentricity and maximum deviation vectors.")
    val dim: Int = ecc.size
    if (K < 2) throw new Exception("invalid number of components")
    if (!ecc.forall(x => x >= 1.0)) throw new Exception("Covariance matrix eccentricity cannot be less than one.")
    ecc.zip(maxDev).map(cov => {
      val mat: DenseMatrix[Double] = DenseMatrix.rand(dim, dim)
      val Q: DenseMatrix[Double] = qr(mat).q
      val eigval: Array[Double] = Random.shuffle(Vector(cov._2 / cov._1, cov._2)
        ++ Vector.fill(K - 2)(Uniform(cov._2 / cov._1, cov._2).draw())).toArray
      val L: DenseMatrix[Double] = diag(new DenseVector(eigval))
      Q.t * L * Q
    })
  }

  // Generate a list of Mean vector
  def genMeanVectors(C: Double, s: Vector[DenseMatrix[Double]]): Vector[DenseVector[Double]] = {
    val K: Int = s.size
    val dim: Int = s.head.rows
    val pso: MGPSO = MGPSO(30, 0.729844, 1.496180, 1.496180)
    val sol = pso.optimize(C, s, 1000)._2
    (0 until K).toVector.map(k => new DenseVector[Double](sol.toArray.slice(dim * k, dim * (k + 1))))
  }

  def Add_outliers(old_data: Vector[Vector[DenseVector[Double]]], m: Vector[DenseVector[Double]], s: Vector[DenseMatrix[Double]]): Vector[DenseVector[Double]] = {
    var data: Vector[Vector[DenseVector[Double]]] = old_data
    val n_outliers: Int = (old_data.size * 0.997).toInt
    val raw: Vector[Array[Double]] = old_data.flatten.map(e => e.data)
    val D: Int = m.head.length

    val maxVal: Array[Double] = raw.tail.foldLeft(raw.head)((curr, cand) => curr.zip(cand).map(e => Math.max(e._1, e._2)))
    val minVal: Array[Double] = raw.tail.foldLeft(raw.head)((curr, cand) => curr.zip(cand).map(e => Math.min(e._1, e._2)))
    var point: Array[Double] = Array.fill(D)(0.0)

    for (n <- 1 to n_outliers) {
      val index: Int = Uniform(0, D - 1).draw().toInt
      if (Bernoulli.distribution(0.5).draw()) {
        point.update(index, maxVal(index))
      } else {
        point.update(index, minVal(index))
      }

      for (i <- 0 until D if i != index) point.update(i, Uniform(minVal(i), maxVal(i)).draw())
      val mdists: Vector[Double] = m.map(c => Math.sqrt((c - new DenseVector(point)).t * (c - new DenseVector(point))))
      val k: Int = mdists.indexOf(mdists.min)
      val ddists: Vector[Double] = data(k).map(c => Math.sqrt((c - new DenseVector(point)).t * (c - new DenseVector(point))))
      val x: Int = ddists.indexOf(ddists.min)
      data = data.updated(k, data(k).updated(x, mutate(data(k)(x), m(k), s(k))))
    }
    data.flatten
  }

  def mutate(p: DenseVector[Double], m: DenseVector[Double], s: DenseMatrix[Double]): DenseVector[Double] = {
    var vec: DenseVector[Double] = p - m
    val norm: Double = math.sqrt(vec.t * vec)
    vec = new DenseVector[Double](vec.data.map(e => e / norm))
    val m_tar: Double = Uniform(5, 9).draw()
    val t: Double = Math.sqrt(Math.pow(m_tar, 2) / (vec.t * inv(s) * vec))
    m + (vec * t)
  }
}