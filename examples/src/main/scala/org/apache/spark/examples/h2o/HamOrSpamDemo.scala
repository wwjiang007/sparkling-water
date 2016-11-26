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

import java.lang.reflect.Field
import java.nio.charset.StandardCharsets

import hex.ModelMetricsBinomial
import hex.deeplearning.DeepLearningModel.DeepLearningParameters
import hex.deeplearning.{DeepLearning, DeepLearningModel}
import org.apache.spark._
import org.apache.spark.h2o._
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}
import sun.misc.Unsafe
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
  // type Vector = Array[Double] // does not work with Spark, catalyst reflection is not good enough
  
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

    val categorizedSMSs = trainingRows map (new CatSMS(_))
    val cutoff = (trainingRows.length * 0.8).toInt
    // Split table
    val (before, after) = trainingRows.splitAt(cutoff)
    val train = buildTable(sc, before)
    val valid = buildTable(sc, after)
//    val inrdd = resultRDD.take(size).toList
    
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
  
  case class SpamModel(msgs: List[String]) {
    val minDocFreq:Int = 4

    lazy val tf: List[Array[Double]] = msgs map weigh
    
    // Build term frequency-inverse document frequency
    lazy val idf0:DocumentFrequencyAggregator = 
      (new DocumentFrequencyAggregator(numFeatures) /: tf)(_ + _)
    
    lazy val modelIdf: Array[Double] = idf0.idf(minDocFreq)

    lazy val weights: List[Array[Double]] = tf map idfNormalize(modelIdf)
    
    def weigh(msg: String): Array[Double] = weighWords(tokenize(msg).toList)

    /** Spam detector */
    def isSpam(sc: SparkContext,
               dlModel: DeepLearningModel)
              (implicit sqlContext: SQLContext, h2oContext: H2OContext) = (msg: String) => {
      import h2oContext.implicits._
      import sqlContext.implicits._
      val weights = weigh(msg)
      val normalizedWeights =idfNormalize(modelIdf)(weights)
      val  hamThreshold: Double = 0.5
      val weighted:RDD[Array[Double]] = toSC(sc, Seq(weights))

      val msgDF = toSC(sc, Seq(DataInside(normalizedWeights))).toDF
      val msgTable: H2OFrame = msgDF
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
    * @param values a term frequency vector
    * @return a TF-IDF vector
    */
  def idfNormalize(idf: Array[Double])(values: Array[Double]): Array[Double] = {
    values zip idf map {case(x,y) => x*y}
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
  
  def mod(i: Int, n: Int) = ((i % n) + n) % n
  
  def arrayFrom(map: Map[Int, Double], size: Int): Array[Double] = {
    0 until size map (i => map.getOrElse(i, 0.0)) toArray
  }

  def weighWords(document: Iterable[String]): Array[Double] = {
    val termFrequencies = scala.collection.mutable.Map.empty[Int, Double]
    document.foreach { term =>
      val i = mod(murmur3(term), numFeatures)
      val count = termFrequencies.getOrElse(i, 0.0) + 1.0
      termFrequencies.put(i, count)
    }

    arrayFrom(termFrequencies.toMap, numFeatures)
  }

  def murmur3(s: String): Int = {
    val seed = 42
    val bytes = s.getBytes(StandardCharsets.UTF_8)
    hashUnsafeBytes(bytes, BYTE_ARRAY_OFFSET, bytes.length, seed)
  }

  private val _UNSAFE: Unsafe = {
    var unsafe: Unsafe = null
    try {
      val unsafeField: Field = classOf[Unsafe].getDeclaredField("theUnsafe")
      unsafeField.setAccessible(true)
       unsafeField.get(null).asInstanceOf[Unsafe]
    }
    catch {
      case cause: Throwable => {
        null
      }
    }
  }
  val BYTE_ARRAY_OFFSET = _UNSAFE.arrayBaseOffset(classOf[Array[Byte]])

  def getInt(x: AnyRef, offset: Long): Int = {
    return _UNSAFE.getInt(x, offset)
  }

  def getByte(x: AnyRef, offset: Long): Byte = {
    return _UNSAFE.getByte(x, offset)
  }

  def hashUnsafeBytes(base: AnyRef, offset: Long, lengthInBytes: Int, seed: Int): Int = {
    assert((lengthInBytes >= 0), "lengthInBytes cannot be negative")
    val lengthAligned: Int = lengthInBytes - lengthInBytes % 4
    var h1: Int = hashBytesByInt(base, offset, lengthAligned, seed)

    for {
      i <- lengthAligned until lengthInBytes
    } {
      val halfWord: Int = getByte(base, offset + i)
      val k1: Int = mixK1(halfWord)
      h1 = mixH1(h1, k1)
    }
    
    fmix(h1, lengthInBytes)
  }

  private def mixH1(h1: Int, k1: Int): Int = {
    val h2 = h1 ^ k1
    val h3 = Integer.rotateLeft(h2, 13)
    val h4 = h3 * 5 + 0xe6546b64
    return h4
  }

  private val C1: Int = 0xcc9e2d51
  private val C2: Int = 0x1b873593

  private def mixK1(k1: Int): Int = {
    val k2 = k1 * C1
    val k3 = Integer.rotateLeft(k2, 15)
    val k4 = k3 * C2
    return k4
  }

  private def fmix (h1: Int, length: Int): Int = {
    val h2 = h1 ^ length
    val h3 = h2 ^ (h2 >>> 16)
    val h4 = h3 * 0x85ebca6b
    val h5 = h4 ^ (h4 >>> 13)
    val h6 = h5 * 0xc2b2ae35
    h6 ^ (h6 >>> 16)
  }

  private def hashBytesByInt(base: AnyRef, offset: Long, lengthInBytes: Int, seed: Int): Int = {
    assert((lengthInBytes % 4 == 0))
    var h1: Int = seed

    for { i <- 0 until lengthInBytes by 4} {
      val halfWord: Int = getInt(base, offset + i)
      val k1: Int = mixK1(halfWord)
      h1 = mixH1(h1, k1)
    }
    
    h1
  }
}

case class CatSMS(target: Int, fv: Array[Double]) {
  def this(sms: TrainingRow) = this("ham"::"spam"::Nil indexOf sms.target, sms.fv)
}

/** Training message representation. */
case class TrainingRow(target: String, fv: Array[Double])

case class DataInside(fv: Array[Double])

/** Document frequency aggregator. */
class DocumentFrequencyAggregator(size: Int) extends Serializable {

  /** number of documents */
  private var m = 0L
  /** document frequency vector */
  private var df: Array[Long] = new Array[Long](size)

  def this() = this(0)

  /** Adds a new frequency vector. */
  def +(doc: Array[Double]): this.type = {
    for { i <- doc.indices } if (doc(i) > 0) df(i) += 1

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
  def idf(minDocFreq: Int): Array[Double] = {
    if (isEmpty) {
      throw new IllegalStateException("Haven't seen any document yet.")
    }
    val inv = df map (x => if (x >= minDocFreq) math.log((m + 1.0) / (x + 1.0)) else 0)

    inv
  }
}
