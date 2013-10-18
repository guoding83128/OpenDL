package org.gd.spark.opendl.example.spark;

import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;
import org.gd.spark.opendl.downpourSGD.SGDTrainConfig;
import org.gd.spark.opendl.downpourSGD.SampleVector;
import org.gd.spark.opendl.downpourSGD.TiedWeightLayer.RBM;
import org.gd.spark.opendl.downpourSGD.train.DownpourSGDTrain;
import org.gd.spark.opendl.example.ClassVerify;
import org.gd.spark.opendl.example.DataInput;

public class RBMTest {
	private static final Logger logger = Logger.getLogger(RBMTest.class);
	
	public static void main(String[] args) {
		try {
			int x_feature = 784;
			int y_feature = 10;
			int n_hidden = 160;
			List<SampleVector> samples = DataInput.readMnist("mnist_784_1000.txt", x_feature, y_feature);
			
			List<SampleVector> trainList = new ArrayList<SampleVector>();
			List<SampleVector> testList = new ArrayList<SampleVector>();
			DataInput.splitList(samples, trainList, testList, 0.7);
			
			JavaSparkContext context = SparkContextBuild.getContext(args);
			JavaRDD<SampleVector> rdds = context.parallelize(trainList);
			rdds.count();
			logger.info("RDD ok.");
			
			RBM rbm = new RBM(x_feature, n_hidden);
            SGDTrainConfig config = new SGDTrainConfig();
            config.setUseCG(true);
            config.setCgEpochStep(50);
            config.setCgTolerance(0);
            config.setCgMaxIterations(10);
            config.setMaxEpochs(50);
            config.setNbrModelReplica(4);
            config.setMinLoss(0.01);
            config.setMrDataStorage(StorageLevel.MEMORY_ONLY());
            config.setPrintLoss(true);
            config.setLossCalStep(3);
            
            logger.info("Start to train RBM.");
            DownpourSGDTrain.train(rbm, rdds, config);
            
            double[] reconstruct_x = new double[x_feature];
            double totalError = 0;
            for(SampleVector test : testList) {
            	rbm.reconstruct(test.getX(), reconstruct_x);
            	totalError += ClassVerify.squaredError(test.getX(), reconstruct_x);
            }
            logger.info("Mean square error is " + totalError / testList.size());
		} catch(Throwable e) {
			logger.error("", e);
		}
	}

}
