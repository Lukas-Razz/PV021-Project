package cz.pv021.neuralnets.layers;

import cz.pv021.neuralnets.functions.OutputFunction;
import java.util.Random;

/**
 * Implementation of output layer.
 * 
 * @author  Lukáš Daubner
 * @since   2016-10-30
 * @version 2016-11-07
 */
public class OutputLayerImpl implements OutputLayer {
    private final OutputFunction outputFunction;
    private final int numberOfUnits;
    private double[] output;
    private Layer upperLayer; // Vstupni
    private double[][] weights;

    public OutputLayerImpl (int numberOfUnits, OutputFunction outputFunction) {
        this.numberOfUnits = numberOfUnits;
        this.output = new double[numberOfUnits];
        this.outputFunction = outputFunction;
    }
    
    @Override
    public void backwardPass () {
        throw new UnsupportedOperationException ("Not supported yet.");
    }
    
    @Override
    public void forwardPass () {
        double[] input = upperLayer.getOutput ();
        double[] innerPotencials = new double[numberOfUnits];
        
        for (int n = 0; n < numberOfUnits; n++) {
            for (int i = 0; i < weights[n].length; i++) {
                innerPotencials[n] += input[i] * weights[n][i];
            }
        }
        
        this.output = outputFunction.apply (innerPotencials);
    }
    
    @Override
    public int getNumberOfUnits () {
        return numberOfUnits;
    }
    
    @Override
    public double[] getOutput () {
        return output;
    }

    @Override
    public void initializeWeights (long seed) {
        Random r = new Random (seed);
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = r.nextGaussian ();
            }
        }
    }
    
    @Override
    public void setUpperLayer (LayerWithOutput layer) {
        this.upperLayer = layer;
        this.weights = new double[numberOfUnits][layer.getNumberOfUnits ()];
    }
}
