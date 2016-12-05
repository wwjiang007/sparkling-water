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

package org.apache.spark

/**
  * This class wraps an iterator and returns number of processed elements instead of number of elements left
  */
class IteratorWithFullSize[T] private(it: Iterator[T]) extends Iterator[T]{
  private var processedElements = 0

  override def hasNext: Boolean = it.hasNext

  override def next(): T = {
    processedElements = processedElements + 1
    it.next()
  }

  override def size: Int = processedElements
}

object IteratorWithFullSize{
  def apply[T](it: Iterator[T]) = new IteratorWithFullSize[T](it)
}
