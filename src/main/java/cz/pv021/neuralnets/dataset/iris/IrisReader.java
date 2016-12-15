package cz.pv021.neuralnets.dataset.iris;

import cz.pv021.neuralnets.dataset.DataReader;
import java.util.ArrayList;
import java.util.List;

/**
 * Reader for Iris data set files.
 * 
 * @author  Lukáš Daubner, Josef Plch
 * @since   2016-12-03
 * @version 2016-12-15
 */
public class IrisReader implements DataReader <IrisExample> {
    private int attributesNo = 0;
    
    @Override
    public List <IrisExample> readDataSet (List <String> lines) {
        List <IrisExample> dataSet = new ArrayList <> ();
        for (String line : lines) {
            IrisExample example = this.readExample (line);
            dataSet.add (example);
        }        
        return dataSet;
    }
    
    /**
     * Read a single example. Desired format:
     * attribute 1,attribute 2,…,attribute n,class
     * 
     * @param string Serialized example.
     * @return Deserialized example.
     */
    @Override
    public IrisExample readExample (String string) {
        String[] columns = string.split (",");
        attributesNo = columns.length - 1;

        double[] attributes = new double [attributesNo];
        for (int i = 0; i < attributesNo; i++) {
            attributes[i] = Double.parseDouble (columns[i]);     
        }

        IrisClass irisClass = IrisClass.read (columns [attributesNo]);
        
        return new IrisExample (attributes, irisClass);
    }
}
