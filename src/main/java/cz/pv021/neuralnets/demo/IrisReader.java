package cz.pv021.neuralnets.demo;

import cz.pv021.neuralnets.utils.Pair;
import static java.lang.Math.max;
import static java.lang.Math.min;
import java.util.ArrayList;
import java.util.List;

/**
 * Reader for Iris data set files.
 * 
 * @author  Lukáš Daubner, Josef Plch
 * @since   2016-12-03
 * @version 2016-12-08
 */
public class IrisReader {
    private int attributesNo = 0;
    
    public List <IrisExample> readDataSet (List <String> lines, boolean normalize) {
        List <IrisExample> dataSet = new ArrayList <> ();
        for (String line : lines) {
            IrisExample example = this.readExample (line);
            dataSet.add (example);
        }
        
        if (normalize) {
            normalize (dataSet);
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
    
    /**
     * Shift the attribute ranges to interval <-1, 1>.
     * 
     * @param dataSet The data set to be modified.
     */
    public void normalize (List <IrisExample> dataSet) {
        for (int att = 0; att < attributesNo; att++) {
            double max = Double.MIN_VALUE;
            double min = Double.MAX_VALUE;
            for (int j = 0; j < dataSet.size(); j++) {
                double value = dataSet.get(j).getAttributes()[att];
                max = max (value, max);
                min = min (value, min);
            }
            double middle = (min + max) / 2;

            // Shift the values to interval <-1, 1>.
            for (int j = 0; j < dataSet.size(); j++) {
                double oldValue = dataSet.get(j).getAttributes()[att];
                double newValue = 2 * (oldValue - middle) / (max - min);
                dataSet.get(j).getAttributes()[att] = newValue;
            }
        }
    }
}
