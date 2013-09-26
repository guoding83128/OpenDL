package org.spark.opendl.downpourSGD.lr;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import org.apache.log4j.Logger;
import org.spark.opendl.downpourSGD.ModelReplicaSplit;
import org.spark.opendl.downpourSGD.SGDTrainConfig;
import org.spark.opendl.downpourSGD.SampleVector;

import spark.api.java.JavaPairRDD;
import spark.api.java.JavaRDD;

/**
 * Logistic Regression(Softmax) train api <p/>
 * 
 * @author GuoDing
 * @since 2013-08-01
 */
public final class LRTrain implements Serializable {
    private static final long serialVersionUID = 1L;
    private static final Logger logger = Logger.getLogger(LRTrain.class);

    /**
     * Standalone multiple threads train work
     * 
     * @param lr The LR node to be trained
     * @param samples The input supervise samples
     * @param config Specify the train configuration
     */
    public static void train(LR lr, List<SampleVector> samples, SGDTrainConfig config) {
        int xy_n = (int) samples.size();
        int nrModelReplica = config.getNbrModelReplica();
        HashMap<Integer, List<SampleVector>> list_map = new HashMap<Integer, List<SampleVector>>();
        for (int i = 0; i < nrModelReplica; i++) {
        	list_map.put(i, new ArrayList<SampleVector>());
        }
        Random rand = new Random(System.currentTimeMillis());
        for (SampleVector v: samples) {
            int id = rand.nextInt(nrModelReplica);
            list_map.get(id).add(v);
        }
        
        List<DeltaThread> threads = new ArrayList<DeltaThread>();
        List<LossThread> loss_threads = new ArrayList<LossThread>();
        for (int i = 0; i < nrModelReplica; i++) {
            threads.add(new DeltaThread(lr, config, list_map.get(i)));
            loss_threads.add(new LossThread(lr));
        }

        // start iteration
        for (int epoch = 1; epoch <= config.getMaxEpochs(); epoch++) {
            // thread start
            for(DeltaThread thread : threads) {
            	thread.train(epoch);
            }

            // waiting for all stop
            while (true) {
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    break;
                }
                boolean allStop = true;
                for(DeltaThread thread : threads) {
                    if (thread.isRunning()) {
                        allStop = false;
                        break;
                    }
                }
                if (allStop) {
                    break;
                }
            }

            // update
            for(DeltaThread thread : threads) {
                lr.mergeParam(thread.getW(), thread.getB(), nrModelReplica);
            }

            logger.info("train done for this iteration-" + epoch);

            if (0 != (epoch % config.getLossCalStep())) {
                continue;
            }

            // sum loss
            for (int i = 0; i < nrModelReplica; i++) {
                loss_threads.get(i).sumLoss(threads.get(i).getX(), threads.get(i).getY());
            }

            // waiting for all stop
            while (true) {
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    break;
                }
                boolean allStop = true;
                for(LossThread thread : loss_threads) {
                    if (thread.isRunning()) {
                        allStop = false;
                        break;
                    }
                }
                if (allStop) {
                    break;
                }
            }

            // sum up
            double totalError = 0;
            for(LossThread thread : loss_threads) {
                totalError += thread.getError();
            }
            totalError /= xy_n;
            logger.info("iteration-" + epoch + " done, total error is " + totalError);
            if (totalError <= config.getMinLoss()) {
                break;
            }
        }
    }

    /**
     * Train the LR with Spark framework
     * 
     * @param lr The LR node to be trained
     * @param samples The input supervise samples
     * @param config Specify the train configuration
     * @param cache Specify whether to store the samples into Spark cache
     */
    public static void train(LR lr, JavaRDD<SampleVector> samples, SGDTrainConfig config, boolean cache) {
        long nbr_xy = samples.count();
        int nrModelReplica = config.getNbrModelReplica();

        // model split
        ModelReplicaSplit<SampleVector> split = new ModelReplicaSplit<SampleVector>();
        JavaPairRDD<Integer, List<SampleVector>> modedSplit = split.split(samples, nrModelReplica, cache);

        // iteration
        for (int epoch = 1; epoch <= config.getMaxEpochs(); epoch++) {
            JavaRDD<DeltaSpark> deltas = modedSplit.map(new DeltaSpark(lr, config, epoch));
            for (DeltaSpark delta: deltas.collect()) {
                lr.mergeParam(delta.getW(), delta.getB(), nrModelReplica);
            }

            logger.info("train done for this iteration-" + epoch);

            if (0 != (epoch % config.getLossCalStep())) {
                continue;
            }

            List<Double> loss_list = modedSplit.map(new LossSpark(lr)).collect();
            double error = 0;
            for (Double loss: loss_list) {
                error += loss;
            }
            error /= nbr_xy;
            logger.info("iteration-" + epoch + " done, total error is " + error);
            if (error <= config.getMinLoss()) {
                break;
            }
        }
    }
}
