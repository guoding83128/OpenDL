/*
 * Copyright 2013 GuoDing
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.gd.spark.opendl.downpourSGD;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.Writer;

/**
 * Persist framework interface for DeepLearning node <p/>
 * 
 * @author GuoDing
 * @since 2013-07-23
 */
public interface SGDPersistable {
	/**
	 * Read in from data inputstream
	 * @param in
	 * @throws IOException
	 */
    public void read(DataInput in) throws IOException;
    
    /**
     * Write out to persistence
     * @param out
     * @throws IOException
     */
    public void write(DataOutput out) throws IOException;
    
    /**
     * Print out just for log, debug
     * @param wr
     * @throws IOException
     */
    public void print(Writer wr) throws IOException;
}
