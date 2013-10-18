package org.gd.spark.opendl.example.standalone;

import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;
import org.gd.spark.opendl.downpourSGD.SGDTrainConfig;
import org.gd.spark.opendl.downpourSGD.SampleVector;
import org.gd.spark.opendl.downpourSGD.Backpropagation.AutoEncoder;
import org.gd.spark.opendl.downpourSGD.train.DownpourSGDTrain;
import org.gd.spark.opendl.example.ClassVerify;
import org.gd.spark.opendl.example.DataInput;

public class dATest {
	private static final Logger logger = Logger.getLogger(dATest.class);
	
	public static void main(String[] args) {
		try {
			int x_feature = 784;
			int y_feature = 10;
			int n_hidden = 160;
			List<SampleVector> samples = DataInput.readMnist("mnist_784_1000.txt", x_feature, y_feature);
			
			List<SampleVector> trainList = new ArrayList<SampleVector>();
			List<SampleVector> testList = new ArrayList<SampleVector>();
			DataInput.splitList(samples, trainList, testList, 0.7);
			
			AutoEncoder da = new AutoEncoder(x_feature, n_hidden);
            SGDTrainConfig config = new SGDTrainConfig();
            config.setUseCG(true);
            config.setDoCorruption(false);
            config.setCorruption_level(0.25);
            config.setCgEpochStep(50);
            config.setCgTolerance(0);
            config.setCgMaxIterations(30);
            config.setMaxEpochs(50);
            config.setNbrModelReplica(4);
            config.setMinLoss(0.01);
            config.setUseRegularization(true);
            config.setPrintLoss(true);
            
            logger.info("Start to train dA.");
            DownpourSGDTrain.train(da, trainList, config);
            
            double[] reconstruct_x = new double[x_feature];
            double totalError = 0;
            for(SampleVector test : testList) {
            	da.reconstruct(test.getX(), reconstruct_x);
            	totalError += ClassVerify.squaredError(test.getX(), reconstruct_x);
            }
            logger.info("Mean square error is " + totalError / testList.size());
		} catch(Throwable e) {
			logger.error("", e);
		}
	}

}
