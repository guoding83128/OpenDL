package org.gd.spark.opendl.example.standalone;

import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;
import org.gd.spark.opendl.downpourSGD.SGDTrainConfig;
import org.gd.spark.opendl.downpourSGD.SampleVector;
import org.gd.spark.opendl.downpourSGD.Backpropagation.BP;
import org.gd.spark.opendl.downpourSGD.train.DownpourSGDTrain;
import org.gd.spark.opendl.example.ClassVerify;
import org.gd.spark.opendl.example.DataInput;

public class BPTest {
	private static final Logger logger = Logger.getLogger(BPTest.class);
	
	public static void main(String[] args) {
		try {
			int x_feature = 784;
			int y_feature = 784;
			List<SampleVector> samples = DataInput.readMnist("mnist_784_1000.txt", x_feature, y_feature);
			
			List<SampleVector> trainList = new ArrayList<SampleVector>();
			List<SampleVector> testList = new ArrayList<SampleVector>();
			DataInput.splitList(samples, trainList, testList, 0.7);
			for(SampleVector v : trainList) {
				for(int i = 0; i < x_feature; i++) {
					v.getY()[i] = v.getX()[i];
				}
			}
			
			int[] hiddens = new int[1];
            hiddens[0] = 160;
            
			BP bp = new BP(x_feature, y_feature, hiddens);
            SGDTrainConfig config = new SGDTrainConfig();
            config.setUseCG(true);
            config.setCgEpochStep(50);
            config.setCgTolerance(0);
            config.setCgMaxIterations(30);
            config.setMaxEpochs(50);
            config.setNbrModelReplica(4);
            config.setMinLoss(0.01);
            config.setUseRegularization(true);
            config.setPrintLoss(true);
            config.setCgInitStepSize(1.0);

            logger.info("Start to train bp.");
            DownpourSGDTrain.train(bp, trainList, config);
            
//            int trueCount = 0;
//            int falseCount = 0;
//            double[] predict_y = new double[y_feature];
//            for(SampleVector test : testList) {
//            	bp.sigmod_output(test.getX(), predict_y);
//            	if(ClassVerify.classTrue(test.getY(), predict_y)) {
//            		trueCount++;
//            	}
//            	else {
//            		falseCount++;
//            	}
//            }
//            logger.info("trueCount-" + trueCount + " falseCount-" + falseCount);
		} catch(Throwable e) {
			logger.error("", e);
		}
	}

}
