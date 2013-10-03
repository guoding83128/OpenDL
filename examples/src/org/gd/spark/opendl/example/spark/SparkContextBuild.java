package org.gd.spark.opendl.example.spark;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.PosixParser;
import org.apache.spark.api.java.JavaSparkContext;

public class SparkContextBuild {
    public static JavaSparkContext getContext(String[] args) throws Exception {
    	Options options = new Options();
    	options.addOption("sparkMasterURL", "smurl", true, "spark master url");
    	options.addOption("sparkJobName", "sjn", false, "my spark job name");
    	options.addOption("sparkHomePath", "shp", true, "specify the spark home path");
    	options.addOption("sparkJobJarPath", "jars", true, "specify the jar search path(directory)");
    	
    	CommandLineParser parser = new PosixParser();
    	CommandLine cmd = parser.parse(options, args);
    	
    	/**
    	 * jar list
    	 */
    	String[] sparkJobJarList = null;
    	String jarsPath = cmd.getOptionValue("sparkJobJarPath");
        File jarDir = new File(jarsPath);
        if (!jarDir.isDirectory()) {
            throw new Exception(jarsPath + " is not a directory!");
        }
        File[] files = jarDir.listFiles();
        List<String> jars = new ArrayList<String>();
        for (File file: files) {
            if (file.isFile() && file.getName().endsWith(".jar")) {
                jars.add(file.getPath());
            }
        }
        sparkJobJarList = new String[jars.size()];
        for (int i = 0; i < sparkJobJarList.length; i++) {
            sparkJobJarList[i] = jars.get(i);
        }
    	
        System.setProperty("spark.local.dir", cmd.getOptionValue("sparkHomePath") + File.separator + "temp");
    	return new JavaSparkContext(cmd.getOptionValue("sparkMasterURL"), 
    			cmd.getOptionValue("sparkJobName", "spark-job-" + System.currentTimeMillis()), 
    			cmd.getOptionValue("sparkHomePath"), 
    			sparkJobJarList, 
    			new HashMap<String,String>());
    }
}
