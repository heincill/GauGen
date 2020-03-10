import spire.math
import breeze.linalg.{DenseMatrix, DenseVector, diag, eigSym, inv, qr}
import breeze.stats.distributions.{Bernoulli, MultivariateGaussian, Uniform}
import spire.math.Interval
import spire.implicits.{DoubleAlgebra, IntAlgebra}

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
    val w: Vector[Double] = n.map(x => x/n.sum)
    w
  }

  // Generate covariance matrices
  def genCovarianceMatrices(dim: Int, ecc: Vector[Double], maxDev: Vector[Double]): Vector[DenseMatrix[Double]] = {
    if (ecc.size != maxDev.size) throw new Exception("Unequal eccentricity and maximum deviation vectors.")
    val K: Int = ecc.size
    if (K < 2) throw new Exception("invalid number of components")
    if (ecc.contains(1.0)) throw new Exception("Covariance matrix eccentricity cannot be less than one.")
    ecc.zip(maxDev).map(cov => {
      if (cov._1 == 1.0) {
        DenseMatrix.eye[Double](dim) * Uniform(0.0, maxDev).draw()
      } else {
        val mat: DenseMatrix[Double] = DenseMatrix.rand(dim, dim)
        val Q: DenseMatrix[Double] = qr(mat).q
        val eigval: Array[Double] = Random.shuffle(Vector(cov._2 / cov._1, cov._2)
          ++ Vector.fill(K-2)(Uniform(cov._2 / cov._1, cov._2).draw())).toArray
        val L: DenseMatrix[Double] = diag(new DenseVector(eigval))
        Q.t * L * Q
      }
    })
  }

  // Generate a list of Mean vector
  def genMeanVectors(C: Double, s: Vector[DenseMatrix[Double]]): Vector[DenseVector[Double]] = {
    val dim: Int = s.head.rows
    val K: Int = s.size
    var m: Array[DenseVector[Double]] = Array(DenseVector.zeros(dim))

    // Generate initial mean vectors
    for (k <- 1 until K) {
      var vec: DenseVector[Double] = new DenseVector[Double](Array.fill(dim)(Uniform(-1.0, 1.0).draw()))
      vec = vec / math.sqrt(vec.t * vec)
      val c: DenseVector[Double] = m(Uniform(0, k-1).draw())
      m = m ++ Vector(c + (vec * Uniform(0, 10).draw()))
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
