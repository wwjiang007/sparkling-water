/*
* Licensed to the Apache Software Foundation (ASF) under one or more
* contributor license agreements.  See the NOTICE file distributed with
* this work for additional information regarding copyright ownership.
* The ASF licenses this file to You under the Apache License, Version 2.0
* (the "License"); you may not use this file except in compliance with
* the License.  You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

package org.apache.spark.examples.h2o

import breeze.linalg.{DenseVector => BDV}
import hex.ModelMetricsBinomial
import hex.deeplearning.DeepLearningModel.DeepLearningParameters
import hex.deeplearning.{DeepLearning, DeepLearningModel}
import org.apache.spark._
import org.apache.spark.h2o._
import org.apache.spark.mllib.feature.{HashingTF, IDFModel}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}
import water.fvec.Vec
import water.support.{H2OFrameSupport, ModelMetricsSupport, SparkContextSupport}

import scala.io.Source
import scala.language.postfixOps
import scala.reflect.ClassTag

/**
 * Demo for NYC meetup and MLConf 2015.
 *
 * It predicts spam text messages.
 * Training dataset is available in the file smalldata/smsData.txt.
 */
object HamOrSpamDemo extends SparkContextSupport with ModelMetricsSupport with H2OFrameSupport{
  val numFeatures = 1024
  
  val DATAFILE="smsData.txt"
  val TEST_MSGS = Seq(
    "Michal, beer tonight in MV?",
    "penis extension, our exclusive offer of penis extension",
    "We tried to contact you re your reply to our offer of a Video Handset? 750 anytime any networks mins? UNLIMITED TEXT?"
  )

  def toSC[X: ClassTag](sc: SparkContext, src: Seq[X]) = sc.parallelize[X](src)

  def main(args: Array[String]) {
    val conf: SparkConf = configure("Sparkling Water Meetup: Ham or Spam (spam text messages detector)")
    // Create SparkContext to execute application on Spark cluster
    val sc = new SparkContext(conf)
    // Initialize H2O context
    implicit val h2oContext = H2OContext.getOrCreate(sc)
    // Initialize SQL context
    implicit val sqlContext = SparkSession.builder().getOrCreate().sqlContext
    
    // Data load
    val lines = readSamples("examples/smalldata/" + DATAFILE)
    val size = lines.size
    val hs = lines map (_(0))
    val msgs = lines map (_(1))
    val spamModel = new SpamModel(msgs)
    
    val trainingRows = hs zip spamModel.weights map TrainingRow.tupled
    val idfModel = spamModel.idfModel

    val categorizedSMSs = trainingRows map (new CatSMS(_))
    val cutoff = (trainingRows.length * 0.8).toInt
    val (before, after) = trainingRows.splitAt(cutoff)
    val train = buildTable(sc, before)
    val valid = buildTable(sc, after)
//    val inrdd = resultRDD.take(size).toList
    
    // Split table
    val keys = Array[String]("train.hex", "valid.hex")
    val ratios = Array[Double](0.8)
    // Build a model
    val dlModel = buildDLModel(train, valid)

    // Collect model metrics
    val trainMetrics = modelMetrics[ModelMetricsBinomial](dlModel, train)
    val validMetrics = modelMetrics[ModelMetricsBinomial](dlModel, valid)
    println(
      s"""
         |AUC on train data = ${trainMetrics.auc}
         |AUC on valid data = ${validMetrics.auc}
       """.stripMargin)

    val isSpam = spamModel.isSpam(sc, dlModel)
    // Detect spam messages
    TEST_MSGS.foreach(msg => {
      println(
        s"""
           |"$msg" is ${if (isSpam(msg)) "SPAM" else "HAM"}
       """.stripMargin)
    })

    // Shutdown Spark cluster and H2O
    h2oContext.stop(stopSparkContext = true)
  }

  def buildTable(sc: SparkContext, trainingRows: List[TrainingRow]): H2OFrame = {
    implicit val h2oContext = H2OContext.getOrCreate(sc)
    import h2oContext.implicits._
    // Initialize SQL context
    implicit val sqlContext = SparkSession.builder().getOrCreate().sqlContext
    import sqlContext.implicits._
    val rdd = toSC(sc, trainingRows)
    val df: DataFrame = rdd.toDF()
    val table: H2OFrame = df
    categorize(table, "target")
    table
  }

  def categorize(table: H2OFrame, colName: String): Unit = {
    val targetVec: Vec = table.vec(colName)
    val categoricalVec: Vec = targetVec.toCategoricalVec
    val targetAt: Int = table.find(colName)
    table.replace(targetAt, categoricalVec).remove()
  }

  def readSamples(dataFile: String): List[Array[String]] = {
    val lines: Iterator[String] = Source.fromFile(dataFile, "ISO-8859-1").getLines()
    val pairs: Iterator[Array[String]] = lines.map(_.split("\t", 2))
    val goodOnes: Iterator[Array[String]] = pairs.filter(!_ (0).isEmpty)
    goodOnes.toList
  }

  val IgnoreWords = Set("the", "not", "for")
  val IgnoreChars = "[,:;/<>\".()?\\-\\\'!01 ]"

  def tokenize(s: String) = {
    var smsText = s.toLowerCase.replaceAll(IgnoreChars, " ").replaceAll("  +", " ").trim
    val words =smsText split " " filter (w => !IgnoreWords(w) && w.length>2)

    words.toSeq
  }

  val hashingTF = new HashingTF(numFeatures)
  
  case class SpamModel(msgs: List[String]) {
    val words = msgs map tokenize
    val minDocFreq:Int = 4
    lazy val tf: List[linalg.Vector] = msgs map weigh
    // Build term frequency-inverse document frequency
    lazy val idf0:DocumentFrequencyAggregator = (new DocumentFrequencyAggregator(numFeatures) /: tf)(_ + _)
    lazy val modelIdf: Vector = idf0.idf(minDocFreq)
    lazy val idfModel = new IDFModel(modelIdf)

    lazy val weights: List[Vector] = tf map idfNormalize(modelIdf)
    
    def weigh(msg: String): Vector = {
      val words = tokenize(msg)
      hashingTF.transform(words)
    }

    /** Spam detector */
    def isSpam(sc: SparkContext,
               dlModel: DeepLearningModel)
              (implicit sqlContext: SQLContext, h2oContext: H2OContext) = (msg: String) => {
      import h2oContext.implicits._
      import sqlContext.implicits._
      val weights = weigh(msg)
      val  hamThreshold: Double = 0.5
      val weighted = sc.parallelize(Seq(weights))
      val msgVector: DataFrame = idfModel.transform(weighted
      ).map(v => VectorInside(v)).toDF
      val msgTable: H2OFrame = msgVector
      val prediction = dlModel.score(msgTable)
      val estimates = prediction.vecs() map (_.at(0)) toList
      val estimate: Double = estimates(1)
      println(s"$msg -> $estimate // $estimates")
      estimate < hamThreshold
    }

  }

  /**
    * Transforms a term frequency (TF) vector to a TF-IDF vector with a IDF vector
    *
    * @param idf an IDF vector
    * @param v a term frequency vector
    * @return a TF-IDF vector
    */
  def idfNormalize(idf: Vector)(v: Vector): Vector = {
    val n = v.size
    v match {
      case SparseVector(size, indices, values) =>
        val newValues = for { (i, v) <-  indices zip values} yield idf(i) * v

        Vectors.sparse(n, indices, newValues)
      case DenseVector(values) =>
        val newValues = new Array[Double](n)
        var j = 0
        while (j < n) {
          newValues(j) = values(j) * idf(j)
          j += 1
        }
        Vectors.dense(newValues)
      case other =>
        throw new UnsupportedOperationException(
          s"Only sparse and dense vectors are supported but got ${other.getClass}.")
    }
  }
  
  /** Builds DeepLearning model. */
  def buildDLModel(train: Frame, valid: Frame,
                   epochs: Int = 10, l1: Double = 0.001,
                   hidden: Array[Int] = Array[Int](200, 200))
                  (implicit h2oContext: H2OContext): DeepLearningModel = {
    import h2oContext.implicits._
    // Build a model
    val dlParams = new DeepLearningParameters()
    dlParams._train = train
    dlParams._valid = valid
    dlParams._response_column = 'target
    dlParams._epochs = epochs
    dlParams._l1 = l1
    dlParams._hidden = hidden

    // Create a job
    val dl = new DeepLearning(dlParams, water.Key.make("dlModel.hex"))
    dl.trainModel.get
  }
}

case class CatSMS(target: Int, fv: mllib.linalg.Vector) {
  def this(sms: TrainingRow) = this("ham"::"spam"::Nil indexOf sms.target, sms.fv)
}

/** Training message representation. */
case class TrainingRow(target: String, fv: mllib.linalg.Vector)

case class VectorInside(fv: mllib.linalg.Vector)

/** Document frequency aggregator. */
class DocumentFrequencyAggregator(size: Int) extends Serializable {

  /** number of documents */
  private var m = 0L
  /** document frequency vector */
  private var df: Array[Long] = new Array[Long](size)

  def this() = this(0)

  /** Adds a new document. */
  def +(doc: Vector): this.type = {
    doc.foreachActive((i,v) => if (v > 0) df(i) += 1)
    m += 1L
    this
  }

  /** Merges another. */
  def merge(other: DocumentFrequencyAggregator): this.type = {
    if (!other.isEmpty) {
      if (isEmpty) {
        df = new Array[Long](other.df.length)
        Array.copy(other.df, 0, df, 0, length = other.df.length)
      } else {
        df.indices foreach (i => df(i) += other.df(i))
      }
      m += other.m
    }
    this
  }

  private def isEmpty: Boolean = m == 0L

  /** Returns the current IDF vector. */
  def idf(minDocFreq: Int): Vector = {
    if (isEmpty) {
      throw new IllegalStateException("Haven't seen any document yet.")
    }
    val inv = df map (x => if (x >= minDocFreq) math.log((m + 1.0) / (x + 1.0)) else 0)
    Vectors.dense(inv)
  }
}
