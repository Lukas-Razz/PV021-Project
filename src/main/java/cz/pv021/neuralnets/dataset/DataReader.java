package cz.pv021.neuralnets.dataset;

import java.util.List;

/**
 * @param <E> Type of examples.
 * 
 * @author  Lukáš Daubner
 * @since   2016-12-10
 * @version 2016-12-15
 */
public interface DataReader <E extends Example> {
    List <E> readDataSet (List <String> lines);

    /**
     * Read a single example.
     * 
     * @param string Serialized example.
     * @return Deserialized example.
     */
    E readExample (String string);
}
