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
package org.gd.spark.opendl.downpourSGD.train;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.gd.spark.opendl.downpourSGD.ModelReplicaSplit;
import org.gd.spark.opendl.downpourSGD.SGDPersistableWrite;
import org.gd.spark.opendl.downpourSGD.SGDTrainConfig;
import org.gd.spark.opendl.downpourSGD.SampleVector;


/**
 * DownpourSGD train framework
 * 
 * @author GuoDing
 * @since 2013-10-05
 */
public final class DownpourSGDTrain implements Serializable {
    private static final long serialVersionUID = 1L;
    private static final Logger logger = Logger.getLogger(DownpourSGDTrain.class);

    /**
     * Standalone multiple threads train work
     * 
     * @param sgd The SGD node to be trained
     * @param samples The input supervise samples
     * @param config Specify the train configuration
     */
    public static void train(SGDBase sgd, List<SampleVector> samples, SGDTrainConfig config) {
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
            threads.add(new DeltaThread(sgd, config, list_map.get(i)));
            loss_threads.add(new LossThread(sgd));
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
            	sgd.mergeParam(thread.getParam(), nrModelReplica);
            }

            logger.info("train done for this iteration-" + epoch);

            /**
             * 1 parameter output
             */
            if(config.isParamOutput() && (0 == (epoch % config.getParamOutputStep()))) {
            	SGDPersistableWrite.output(config.getParamOutputPath(), sgd);
            }
            
            /**
             * 2 loss print
             */
            if(!config.isPrintLoss()) {
            	continue;
            }
            if (0 != (epoch % config.getLossCalStep())) {
                continue;
            }

            // sum loss
            for (int i = 0; i < nrModelReplica; i++) {
            	loss_threads.get(i).sumLoss(threads.get(i).getSamples());
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
     * DownpourSGD train with Spark framework
     * 
     * @param sgd The SGD node to be trained
     * @param samples The input supervise samples
     * @param config Specify the train configuration
     * @param cache Specify whether to store the samples into Spark cache
     */
    public static void train(SGDBase sgd, JavaRDD<SampleVector> samples, SGDTrainConfig config) {
        long nbr_xy = samples.count();
        int nrModelReplica = config.getNbrModelReplica();

        // model split
        ModelReplicaSplit<SampleVector> split = new ModelReplicaSplit<SampleVector>();
        JavaPairRDD<Integer, List<SampleVector>> modedSplit = split.split(samples, nrModelReplica, config);

        // iteration
        for (int epoch = 1; epoch <= config.getMaxEpochs(); epoch++) {
        	logger.info("start train for this iteration-" + epoch);
            JavaRDD<DeltaSpark> deltas = modedSplit.map(new DeltaSpark(sgd, config, epoch));
            for (DeltaSpark delta: deltas.collect()) {
            	sgd.mergeParam(delta.getParam(), nrModelReplica);
            }
            logger.info("train done for this iteration-" + epoch);

            /**
             * 1 parameter output
             */
            if(config.isParamOutput() && (0 == (epoch % config.getParamOutputStep()))) {
            	SGDPersistableWrite.output(config.getParamOutputPath(), sgd);
            }
            
            /**
             * 2 loss print
             */
            if(!config.isPrintLoss()) {
            	continue;
            }
            if (0 != (epoch % config.getLossCalStep())) {
                continue;
            }

            List<Double> loss_list = modedSplit.map(new LossSpark(sgd)).collect();
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
