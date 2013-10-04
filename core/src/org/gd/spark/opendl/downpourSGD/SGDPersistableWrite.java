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
