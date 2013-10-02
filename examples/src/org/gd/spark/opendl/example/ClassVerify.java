package org.gd.spark.opendl.example;

public class ClassVerify {
    public static boolean classTrue(double[] y, double[] predict_y) {
    	if(y.length != predict_y.length) {
    		return false;
    	}
    	double max = -1;
    	int y_idx = -1;
    	int predict_y_idx = -1;
    	for(int i = 0; i < y.length; i++) {
    		if(y[i] > max) {
    			max = y[i];
    			y_idx = i;
    		}
    	}
    	max = -1;
    	for(int i = 0; i < predict_y.length; i++) {
    		if(predict_y[i] > max) {
    			max = predict_y[i];
    			predict_y_idx = i;
    		}
    	}
    	return (y_idx == predict_y_idx);
    }
    
    public static String printDoubleArray(double[] d) {
    	StringBuffer sb = new StringBuffer();
    	for(int i = 0; i < d.length; i++) {
    		sb.append(d[i]);
    		sb.append(",");
    	}
    	return sb.toString();
    }
}
