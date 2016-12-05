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

package org.apache.spark.h2o.converters

import org.apache.spark.h2o.utils.ReflectionUtils._
import org.apache.spark.h2o.utils.SupportedTypes
import org.apache.spark.h2o.utils.SupportedTypes._
import org.apache.spark.unsafe.types.UTF8String

import scala.language.postfixOps

/**
  * Methods which each ReadConverterContext has to implement.
  *
  * Read Converter Context is a class which holds the state of connection/chunks and allows us to read/download data from those chunks
  * via unified API
  */
trait ReadConverterContext {

  /** Type from which we query the data */
  type DataSource

  /** Key pointing to H2OFrame containing the data */
  val keyName: String

  /** Chunk Idx/Partition index */
  val chunkIdx: Int

  /** Current row index */
  var rowIdx: Int = 0

  def numRows: Int
  def increaseRowIdx() = rowIdx += 1

  def hasNext = rowIdx < numRows

  type OptionReader = Int => Option[Any]

  type Reader = Int => Any

  /**
    * For a given array of source column indexes and required data types,
    * produces an array of value providers.
    *
    * @param columnIndexesWithTypes lists which columns we need, and what are the required types
    * @return an array of value providers. Each provider gives the current column value
    */
  def columnValueProviders(columnIndexesWithTypes: Array[(Int, SimpleType[_])]): Array[() => Option[Any]] = {
    for {
      (columnIndex, supportedType) <- columnIndexesWithTypes
      reader = OptionReaders(byBaseType(supportedType))
      provider = () => reader.apply(columnIndex)
    } yield provider
  }

  /**
    * Returns Option with data was successfully obtained or none otherwise
    */
  protected def returnOption[T](read: DataSource => T)(columnNum: Int): Option[T]

  /**
    * Returns the data if it was obtained successfully or the default value
    */
  protected def returnSimple[T](ifMissing: String => T, read: DataSource => T)(columnNum: Int): T

  protected def longAt(source: DataSource): Long
  protected def doubleAt(source: DataSource): Double
  protected def booleanAt(source: DataSource) = longAt(source) == 1
  protected def byteAt(source: DataSource) = longAt(source).toByte
  protected def intAt(source: DataSource) = longAt(source).toInt
  protected def shortAt(source: DataSource) = longAt(source).toShort
  protected def floatAt(source: DataSource) = longAt(source).toFloat
  // TODO(vlad): take care of this bad typing
  protected def timestamp(source: DataSource) = longAt(source) * 1000L
  protected def string(source: DataSource): String
  // TODO(vlad): check if instead of stringification, we could use bytes
  protected def utfString(source: DataSource) = UTF8String.fromString(string(source))

  /**
    * This map registers for each type corresponding extractor
    *
    * Given a a column number, returns an Option[T]
    * with the value parsed according to the type.
    * You can override it.
    *
    * A map from type name to option reader
    */
  protected lazy val ExtractorsTable: Map[SimpleType[_], DataSource => _] = Map(
    Boolean    -> booleanAt _,
    Byte       -> byteAt _,
    Double     -> doubleAt _,
    Float      -> floatAt _,
    Integer    -> intAt _,
    Long       -> longAt _,
    Short      -> shortAt _,
    String     -> string _,
    UTF8       -> utfString _,
    Timestamp  -> timestamp _
  )

  private lazy val OptionReadersMap: Map[OptionalType[_], OptionReader] =
    ExtractorsTable map {
      case (t, reader) => SupportedTypes.byBaseType(t) -> returnOption(reader) _
    } toMap

  private lazy val SimpleReadersMap: Map[SimpleType[_], Reader] =
    ExtractorsTable map {
      case (t, reader) => t -> returnSimple(t.ifMissing, reader) _
    } toMap

  private lazy val OptionReaders: Map[OptionalType[_], OptionReader] = OptionReadersMap withDefault
    (t => throw new scala.IllegalArgumentException(s"Type $t conversion is not supported in Sparkling Water"))

  private lazy val SimpleReaders: Map[SimpleType[_], Reader] = SimpleReadersMap withDefault
    (t => throw new scala.IllegalArgumentException(s"Type $t conversion is not supported in Sparkling Water"))

  lazy val readerMapByName: Map[NameOfType, Reader] = (OptionReaders ++ SimpleReaders) map {
    case (supportedType, reader) => supportedType.name -> reader
  } toMap


}
