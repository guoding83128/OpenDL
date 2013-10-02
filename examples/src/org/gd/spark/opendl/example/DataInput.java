package org.gd.spark.opendl.example;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.gd.spark.opendl.downpourSGD.SampleVector;

public class DataInput {
	private static Random rand = new Random(System.currentTimeMillis());
	
	/**
	 * Read sample data from mnist_784 text file 
	 * @param path
	 * @return
	 * @throws Exception
	 */
	public static List<SampleVector> readMnist(String path, int x_feature, int y_feature) throws Exception {
		List<SampleVector> ret = new ArrayList<SampleVector>();
		String str = null;
        BufferedReader br = new BufferedReader(new FileReader(path));
        while (null != (str = br.readLine())) {
            String[] splits = str.split(",");
            SampleVector xy = new SampleVector(x_feature, y_feature);
            xy.getY()[Integer.valueOf(splits[0])] = 1;
            for (int i = 1; i < splits.length; i++) {
                xy.getX()[i - 1] = Double.valueOf(splits[i]);
            }
            ret.add(xy);
        }
        br.close();
		return ret;
	}

	/**
	 * Parallelize list to RDD
	 * @param context
	 * @param list
	 * @return
	 * @throws Exception
	 */
	public static JavaRDD<SampleVector> toRDD(JavaSparkContext context, List<SampleVector> list) throws Exception {
		return context.parallelize(list);
	}
	
	/**
	 * Split total list read from file to train and test part
	 * @param totalList
	 * @param trainList
	 * @param testList
	 * @param trainRatio
	 */
	public static void splitList(List<SampleVector> totalList, List<SampleVector> trainList, List<SampleVector> testList, double trainRatio) {
		for (SampleVector sample : totalList) {
			if (rand.nextDouble() <= trainRatio) {
				trainList.add(sample);
			}
			else {
				testList.add(sample);
			}
		}
	}
}
