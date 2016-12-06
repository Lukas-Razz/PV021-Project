package cz.pv021.neuralnets.demo;

import cz.pv021.neuralnets.utils.Pair;
import static java.lang.Math.max;
import static java.lang.Math.min;
import java.util.ArrayList;
import java.util.List;

/**
 * Reader for Iris data set files.
 * 
 * @author  Josef Plch
 * @since   2016-12-03
 * @version 2016-12-06
 */
public class IrisReader {
    private int recordsNo = 0;
    private int attributesNo = 0;
    
    public Pair <double[], IrisClass> readEntry (String string) {
        String[] columns = string.split (",");
        attributesNo = columns.length - 1;

        double[] attributes = new double [attributesNo];
        for (int i = 0; i < attributesNo; i++) {
            attributes[i] = Double.parseDouble (columns[i]);     
        }

        IrisClass irisClass = IrisClass.read (columns [attributesNo]);
        
        return new Pair <> (attributes, irisClass);
    }
    
    public List<Pair<double[], IrisClass>> getDataSet(List<String> data, boolean normalize) {
        List<Pair<double[], IrisClass>> list = new ArrayList<>();
        for (String line : data) {
            Pair <double[], IrisClass> entry = this.readEntry (line);
            list.add(entry);
            recordsNo++;
        }
        if(normalize) {
            for(int i = 0; i<attributesNo; i++) {
                double max = Double.MIN_VALUE;
                double min = Double.MAX_VALUE;
                for(int j = 0; j<list.size(); j++) {
                    max = max(list.get(j).getA()[i], max);
                    min = min(list.get(j).getA()[i], min);
                }
                for(int j = 0; j<list.size(); j++) {
                    list.get(j).getA()[i] = 2/(max-min) * (list.get(j).getA()[i] - ((min+max)/2));
                }
            }
        }
        return list;
    }
}
