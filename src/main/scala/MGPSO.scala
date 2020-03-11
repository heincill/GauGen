import breeze.linalg.eigSym.EigSym
import breeze.linalg.{DenseMatrix, DenseVector, eigSym}

/**
 * Mean generation PSO
 *
 * @param swarm_size
 * @param w
 * @param c1
 * @param c2
 */
case class MGPSO(swarm_size: Int, w: Double, c1: Double, c2: Double) {

  private var position: Vector[Vector[Double]] = _
  private var velocity: Vector[Vector[Double]] = _

  private var pbest_position: Vector[Vector[Double]] = _
  private var pbest_cost: Vector[Double] = _

  private var best_position: Vector[Double] = _
  private var best_cost: Double = _

  private var m: Int = _
  private var dim: Int = _
  private var K: Int = _
  private var C: Double = _
  private var s: Vector[DenseMatrix[Double]] = _

  def optimize(C: Double, s: Vector[DenseMatrix[Double]], iterations: Int): (Double, Vector[Double]) = {
    this.dim = s.head.rows
    this.K = s.size
    this.C = C
    this.s = s
    m = dim * K
    initialize()

    (1 to iterations).foreach(_ => {
      update_velocity()
      update_position()
      update_pbest()
      update_best()
    })

    (best_cost, best_position)
  }

  private def initialize(): Unit = {
    position = Vector.tabulate(swarm_size)(_ => Vector.tabulate(m)(_ => Math.random()))
    velocity = Vector.fill(swarm_size)(Vector.fill(m)(0.0))

    pbest_position = position
    pbest_cost = pbest_position.map(p => f(p))

    val i: Int = argmin(pbest_cost)
    best_position = pbest_position(i)
    best_cost = pbest_cost(i)
  }

  private def update_position(): Unit =
    position = (0 until swarm_size).toVector.map(i =>
      (0 until m).toVector.map(j =>
        position(i)(j) + velocity(i)(j)))

  private def update_velocity(): Unit =
    velocity = (0 until swarm_size).toVector.map(i =>
      (0 until m).toVector.map(j =>
        w * velocity(i)(j) +
          c1 * Math.random() * (best_position(j) - position(i)(j)) +
          c2 * Math.random() * (pbest_position(i)(j) - position(i)(j))))

  private def update_pbest(): Unit =
    (0 until swarm_size).foreach(i => {
      val g: Double = f(position(i))
      if (g < pbest_cost(i)) {
        pbest_cost = pbest_cost.updated(i, g)
        pbest_position = pbest_position.updated(i, position(i))
      }
    })

  private def update_best(): Unit =
    (0 until swarm_size).foreach(i => {
      if (pbest_cost(i) < best_cost) {
        best_cost = pbest_cost(i)
        best_position = pbest_position(i)
      }
    })

  def f(x: Vector[Double]): Double = {
    val means: Vector[DenseVector[Double]] = (0 until K).toVector.map(k => new DenseVector[Double](x.toArray.slice(dim * k, dim * (k+1))))
    math.abs(C - (0 until (K-1)).flatMap(k => ((k+1) until K).map(l => sep(means(k), s(k), means(l), s(l)))).sum / (K * (K-1) / 2))
  }

  def sep(mu_k: DenseVector[Double], sig_k: DenseMatrix[Double], mu_l: DenseVector[Double], sig_l: DenseMatrix[Double]): Double = {
    val v: DenseVector[Double] = mu_k - mu_l
      math.sqrt(v.t * v) / math.sqrt(dim * math.max(eigSym(sig_k).eigenvalues.data.max, eigSym(sig_l).eigenvalues.data.max))
  }

  private def argmax(list: Vector[Double]): Int = list.indexOf(list.max)
  private def argmin(list: Vector[Double]): Int = list.indexOf(list.min)
}
