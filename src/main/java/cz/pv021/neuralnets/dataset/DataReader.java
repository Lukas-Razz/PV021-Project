package cz.pv021.neuralnets.dataset;

import cz.pv021.neuralnets.dataset.iris.IrisExample;
import java.util.List;

/**
 * @author  Lukáš Daubner
 * @since   2016-12-10
 * @version 2016-12-10
 */
public interface DataReader {

    List<Example> readDataSet(List<String> lines);

    /**
     * Read a single example. Desired format:
     * attribute 1,attribute 2,…,attribute n,class
     *
     * @param string Serialized example.
     * @return Deserialized example.
     */
    IrisExample readExample(String string);
    
}
