import spire.math
import breeze.linalg.{DenseMatrix, DenseVector, diag, eigSym, inv, qr}
import breeze.stats.distributions.{MultivariateGaussian, Bernoulli}
import spire.math.Interval
import spire.implicits.{DoubleAlgebra, IntAlgebra}
import scalaz.Scalaz._

object GaussianMixture {
  /*  D -> Dimensions
      K -> # Clusters
      N -> # Samples
      C -> Separability between clusters
      O -> # Outliers
      S -> Degree of symmetry
   */

  def GenGMM(N: Int, w: Vector[Double], m: Vector[DenseVector[Double]], s: Vector[DenseMatrix[Double]]): Vector[DenseVector[Double]] = {
    val clust_n: Vector[Int] = w.map(e => Math.round(e * N).toInt)
    clust_n.zip(m).zip(s).map(p => Array.fill(p._1._1)(MultivariateGaussian(p._1._2, p._2).draw()))
      .foldLeft(Array[DenseVector[Double]]())(_ ++ _).toVector
  }

  def GenGMM(K: Int, D: Int, R: Double, N: Int, C: Double, O: Boolean, S: Boolean): Vector[DenseVector[Double]] = {
    // Generate weights for GMM
    val w: Vector[Double] = genWeights(K, R)

    // Generate samples for each gaussian
    val clust_samples: Vector[Int] = w.map(e => Math.round(e * N).toInt)

    // Generate K (D x D) Covariance matrices with given symmetry
    val s: Vector[DenseMatrix[Double]] = genCovarianceMatrices(S, K, D)

    // Separation
    val m: Vector[DenseVector[Double]] = genMeanVectors(D, K, C, s)

    // Generate list of gaussian data
    val pre_data: Vector[Vector[DenseVector[Double]]] = (0 until K).map(k => Vector.fill(clust_samples(k))(MultivariateGaussian(m(k), s(k)).draw())).toVector

    // Flatten gaussian data
    var data: Vector[DenseVector[Double]] = pre_data.flatten

    // Add outliers
    if (O) data = Add_outliers(pre_data, m, s)
    data
  }

  // Generate a list of mixture coefficients
  def genWeights(k: Int, r: Double): Vector[Double] = {
    // Allocate a random weight to each cluster
    val n: Vector[Double] = Vector(1, r) ++ Vector.fill(k - 2)(breeze.stats.distributions.Uniform(1, r).draw())
    val w: Vector[Double] = n.map(x => x/n.sum)
    w
  }

  // Generate covariance matrices
  def genCovarianceMatrices(S: Boolean, K: Int, D: Int): Vector[DenseMatrix[Double]] = {
    if (S) {
      Vector.fill(K)(DenseMatrix.eye[Double](D) * Dist.uniform(Interval(0.5, 20.0)).eval(RNG.fromTime))
    } else {
      Vector.fill(K)({
        val mat: DenseMatrix[Double] = DenseMatrix.rand(D, D)
        val Q: DenseMatrix[Double] = qr(mat).q
        val L: DenseMatrix[Double] = diag(new DenseVector(Array.fill(D)(cilib.Dist.uniform(Interval(0.5, 20.0)).eval(RNG.fromTime))))
        Q.t * L * Q
      })
    }
  }

  // Generate a list of Mean vector
  def genMeanVectors(D: Int, K: Int, C: Double, s: Vector[DenseMatrix[Double]]): Vector[DenseVector[Double]] = {
    var m: Array[DenseVector[Double]] = Array(DenseVector.zeros(D))

    // Generate initial mean vectors
    for (k <- 1 until K) {
      var vec: DenseVector[Double] = new DenseVector[Double](Array.fill(D)(Dist.uniform(Interval(-1.0, 1.0)).eval(RNG.fromTime)))
      vec = vec / math.sqrt(vec.t * vec)
      val c: DenseVector[Double] = m(Dist.uniformInt(Interval(0, k - 1)).eval(RNG.fromTime))
      m = m ++ Vector(c + (vec * Dist.uniform(Interval(0.0, 10.0)).eval(RNG.fromTime)))
    }

    // Calculate the separation of 2 mean vectors
    val sepa: (Int, Int) => Double = (k1: Int, k2: Int) =>
      C * math.sqrt(D * math.max(eigSym(s(k1)).eigenvalues.data.max, eigSym(s(k2)).eigenvalues.data.max))

    //val min_sep: Array[Array[Double]] = (0 until (K - 1)).map(k1 => ((k1 + 1) until K).map(k2 => sepa(k1, k2)).toArray).toArray
    var min_sep: Array[Array[Double]] = (0 until K).map(k1 => (0 until K).map(k2 => sepa(k1, k2)).toArray).toArray
    var dists: Array[Array[Double]] = (0 until K).map(k1 => (0 until K).map(k2 => LASH.E_dist(m(k1), m(k2))).toArray).toArray
    var sep: Array[Array[Double]] = dists.zip(min_sep).map(e => e._1.zip(e._2).map(el => el._2 - el._1))
    for (i <- 0 until K) sep(i).update(i, 0.0)

    var iter: Int = 0
    while (sep.flatten.max > 0.0) {
      var k1: Int = 0
      while (k1 < (K - 1)) {
        var k2: Int = k1 + 1
        while (k2 < K) {
          if (sep(k1)(k2) > 0.0) {
            if (Dist.uniform(Interval(0.0, 1.0)).eval(RNG.fromTime) < 0.5) {
              var vec: DenseVector[Double] = m(k2) - m(k1)  // Get vector between k1 and k2
              vec = vec / math.sqrt(vec.t * vec)            // Normalize the vector
              m.update(k2, m(k1) + (vec * min_sep(k1)(k2))) // Update k2
            } else {
              var vec: DenseVector[Double] = m(k1) - m(k2)  // Get vector between k1 and k2
              vec = vec / math.sqrt(vec.t * vec)            // Normalize the vector
              m.update(k1, m(k2) + (vec * min_sep(k1)(k2))) // Update k2
            }

            // update dists and separations
            dists = (0 until K).map(k1 => (0 until K).map(k2 => LASH.E_dist(m(k1), m(k2))).toArray).toArray
            sep = dists.zip(min_sep).map(e => e._1.zip(e._2).map(el => el._2 - el._1))
            for (i <- 0 until K) sep(i).update(i, 0.0)
          }
          k2 += 1
        }
        k1 += 1
      }
      iter += 1
      if (iter % 500 == 0) {
        m = Array(DenseVector.zeros(D))

        // Generate initial mean vectors
        for (k <- 1 until K) {
          var vec: DenseVector[Double] = new DenseVector[Double](Array.fill(D)(Dist.uniform(Interval(-1.0, 1.0)).eval(RNG.fromTime)))
          vec = vec / math.sqrt(vec.t * vec)
          val c: DenseVector[Double] = m(Dist.uniformInt(Interval(0, k - 1)).eval(RNG.fromTime))
          m = m ++ List(c + (vec * Dist.uniform(Interval(0.0, 10.0)).eval(RNG.fromTime)))
        }

        min_sep = (0 until K).map(k1 => (0 until K).map(k2 => sepa(k1, k2)).toArray).toArray
        dists = (0 until K).map(k1 => (0 until K).map(k2 => LASH.E_dist(m(k1), m(k2))).toArray).toArray
        sep = dists.zip(min_sep).map(e => e._1.zip(e._2).map(el => el._2 - el._1))
        for (i <- 0 until K) sep(i).update(i, 0.0)
        iter = 0
      }
    }
    m.toVector
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
      val index: Int = Dist.uniformInt(Interval(0, D - 1)).eval(RNG.fromTime)
      if (Bernoulli.distribution(0.5).draw()) {
        point.update(index, maxVal(index))
      } else {
        point.update(index, minVal(index))
      }
      for (i <- 0 until D if i != index) point.update(i, Dist.uniform(Interval(minVal(i), maxVal(i))).eval(RNG.fromTime))
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
    val m_tar: Double = Dist.uniform(Interval(5, 9)).eval(RNG.fromTime)
    val t: Double = Math.sqrt(Math.pow(m_tar, 2) / (vec.t * inv(s) * vec))
    m + (vec * t)
  }
