package cz.pv021.neuralnets.demo;

import cz.pv021.neuralnets.utils.Pair;

/**
 * Reader for Iris data set files.
 * 
 * @author  Josef Plch
 * @since   2016-12-03
 * @version 2016-12-04
 */
public class IrisReader {
    public Pair <double[], IrisClass> readEntry (String string) {
        String[] columns = string.split (",");
        int numberOfAttributes = columns.length - 1;

        double[] attributes = new double [numberOfAttributes];
        for (int i = 0; i < numberOfAttributes; i++) {
            attributes[i] = Double.parseDouble (columns[i]);
        }

        IrisClass irisClass = IrisClass.read (columns [numberOfAttributes]);
        
        return new Pair <> (attributes, irisClass);
    }
}
