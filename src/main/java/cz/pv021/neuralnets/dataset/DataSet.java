package cz.pv021.neuralnets.dataset;

import cz.pv021.neuralnets.utils.Round;
import static java.lang.Math.max;
import static java.lang.Math.min;
import java.util.ArrayList;
import java.util.List;

/**
 * @param <C> Type of data class.
 * @param <E> Type of examples.
 * 
 * @author  Lukáš Daubner
 * @since   2016-12-10
 * @version 2016-12-10
 */
public class DataSet <C extends DataClass, E extends SimpleExample <C>> {
    List <E> trainSet;
    List <E> testSet;
    
    int numberOfAttributes;
    
    final double roundBy = 1000.0;
    
    public DataSet (List <E> examples, double splitFactor) {
        trainSet = new ArrayList<>();
        testSet = new ArrayList<>();
        
        int testSetSamples = (int) Math.floor(examples.size() * splitFactor);
        for(int i=0; i<testSetSamples; i++) {
            testSet.add(examples.get(i));
        }
        for(int i=testSetSamples; i<examples.size(); i++) {
            trainSet.add(examples.get(i));
        }
        
        numberOfAttributes = examples.get(0).getAttributes().length;
    }
    
    public DataSet (List <E> trainSet, List <E> testSet) {
        this.trainSet = trainSet;
        this.testSet = testSet;
        
        numberOfAttributes = trainSet.get(0).getAttributes().length;
    }

    public List <E> getTestSet() {
        return testSet;
    }

    public List <E> getTrainSet() {
        return trainSet;
    }
    
    /**
     * Shift the attribute ranges to interval <-1, 1>.
     * 
     */
    public void normalizeToMinusOnePlusOne () {
        int totalExamples = testSet.size() + trainSet.size();
        
        for (int att = 0; att < numberOfAttributes; att++) {
            double max = Double.MIN_VALUE;
            double min = Double.MAX_VALUE;
            for (int j = 0; j < totalExamples; j++) {
                double value = getExampleByIndex(j).getAttributes()[att];
                max = max (value, max);
                min = min (value, min);
            }
            double middle = (min + max) / 2;

            // Shift the values to interval <-1, 1>.
            for (int j = 0; j < totalExamples; j++) {
                double oldValue = getExampleByIndex(j).getAttributes()[att];
                double newValue = 2 * (oldValue - middle) / (max - min);
                getExampleByIndex(j).getAttributes()[att] = Round.Round(newValue, roundBy);
            }
        }
    }
    
    /**
     * Shift the attribute ranges to interval <-1, 1>.
     * 
     */
    public void normalizeToZeroPlusOne () {
        int totalExamples = testSet.size() + trainSet.size();
        
        for (int att = 0; att < numberOfAttributes; att++) {
            double max = Double.MIN_VALUE;
            double min = Double.MAX_VALUE;
            for (int j = 0; j < totalExamples; j++) {
                double value = getExampleByIndex(j).getAttributes()[att];
                max = max (value, max);
                min = min (value, min);
            }

            // Shift the values to interval <-1, 1>.
            for (int j = 0; j < totalExamples; j++) {
                double oldValue = getExampleByIndex(j).getAttributes()[att];
                double newValue = (oldValue - min) / (max - min);
                getExampleByIndex(j).getAttributes()[att] = Round.Round(newValue, roundBy);
            }
        }
    }
    
    private E getExampleByIndex (int i) {
        if (i < trainSet.size()) {
            return trainSet.get(i);
        }
        else {
            return testSet.get(i - trainSet.size());
        }
    }
    
    public List <List <E>> splitToBatch (int batchSize) {
        List <List <E>> batches = new ArrayList<>();
        
        int totalBatches = (int) Math.ceil(trainSet.size() / (double)batchSize);
        int index = 0;
        for(int i=0; i<totalBatches; i++) {
            List<E> batch = new ArrayList<>();
            for(int j=0; j<batchSize; j++) {
                if(index < trainSet.size()) {
                    batch.add(trainSet.get(index));
                    index ++;
                }
                else {
                    break;
                }
            }
            batches.add(batch);
        }
        return batches;
    }
}
