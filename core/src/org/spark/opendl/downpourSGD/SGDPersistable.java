package org.spark.opendl.downpourSGD;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.Writer;

/**
 * Persist framework interface for DeepLearning node <p/>
 * 
 * @author GuoDing
 * @since 2013-07-23
 */
public interface SGDPersistable {
    public void read(DataInput in) throws IOException;
    public void write(DataOutput out) throws IOException;
    public void print(Writer wr) throws IOException;
}
