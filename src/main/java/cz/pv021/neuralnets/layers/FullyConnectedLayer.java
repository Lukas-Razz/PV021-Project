package cz.pv021.neuralnets.layers;

import cz.pv021.neuralnets.functions.ActivationFunction;
import java.util.Random;

/**
 * @author  Lukáš Daubner
 * @since   2016-10-30
 * @version 2016-11-08
 */
public class FullyConnectedLayer implements HiddenLayer {
    private final ActivationFunction activationFunction;
    private LayerWithInput lowerLayer; // Vystupni
    private LayerWithOutput upperLayer; // Vstupni
    private final int numberOfUnits;
    private final double[] output;
    private double[][] weights;

    public FullyConnectedLayer (int numberOfUnits, ActivationFunction activationFunction) {
        this.numberOfUnits = numberOfUnits;
        this.output = new double[numberOfUnits];
        this.activationFunction = activationFunction;
    }

    @Override
    public void backwardPass () {
        throw new UnsupportedOperationException ("Not supported yet.");
    }

    @Override
    public void forwardPass () {
        double[] input = upperLayer.getOutput ();
        for (int n = 0; n < numberOfUnits; n++) {
            // int innerPotencial = 0;
            double innerPotencial = 0;
            for (int i = 0; i < weights[n].length; i++) {
                innerPotencial += input[i] * weights[n][i];
            }
            output[n] = activationFunction.apply (innerPotencial);
        }
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
    public void setLowerLayer (LayerWithInput nextLayer) {
        this.lowerLayer = nextLayer;
    }
    
    @Override
    public void setUpperLayer (LayerWithOutput previousLayer) {
        this.upperLayer = previousLayer;
        weights = new double[numberOfUnits][previousLayer.getNumberOfUnits ()];
    }
}
