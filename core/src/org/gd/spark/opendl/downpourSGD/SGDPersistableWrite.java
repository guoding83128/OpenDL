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

import java.io.DataOutputStream;
import java.io.FileOutputStream;

import org.apache.log4j.Logger;

public class SGDPersistableWrite {
	private static final Logger logger = Logger.getLogger(SGDPersistableWrite.class);
	
    public static void output(String path, SGDPersistable sgd) {
    	DataOutputStream dos = null;
    	try {
    		dos = new DataOutputStream(new FileOutputStream(path));
    		sgd.write(dos);
    	} catch(Throwable e) {
    		logger.error(path, e);
    	}
    	if(null != dos) {
    		try {
				dos.close();
			} catch (Throwable e) {
			}
    	}
    }
}
