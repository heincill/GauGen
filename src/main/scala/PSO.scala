case class PSO(swarm_size: Int, w: Double, c1: Double, c2: Double) {

  private var position: Vector[Vector[Double]] = _
  private var velocity: Vector[Vector[Double]] = _

  private var pbest_pos: Vector[Vector[Double]] = _
  private var pbest_cost: Vector[Double] = _

  private var best_pos: Vector[Double] = _
  private var best_cost: Double = _

  def optimize(f: Vector[Double] => Double, iterations: Int/*, upper_bound: Vector[Double], lower_bound: Vector[Double]*/): (Double, Vector[Double]) = {

  }

  private def initialize(): Unit = {

  }

  private def update_position(): Unit = {

  }

  private def update_velocity(): Unit = {

  }

  private def update_pbest(): Unit = {

  }

  private def update_best(): Unit = {

  }
}
