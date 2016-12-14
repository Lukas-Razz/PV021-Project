package cz.pv021.neuralnets.layers;

import java.util.Arrays;
import java.util.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import cz.pv021.neuralnets.functions.HiddenFunction;

/**
 * A recursive layer, i.e. layer with self-loops.
 * 
 * @author  Lukáš Daubner, Josef Plch
 * @since   2016-10-30
 * @version 2016-12-14
 */
public class FullyConnectedRecursiveLayer extends FullyConnectedLayer implements RecursiveHiddenLayer {
    private final Logger logger = LoggerFactory.getLogger (FullyConnectedRecursiveLayer.class);
    
    private int id;
    
    private double[][] loopWeights;
    private final int layerSize;

    public FullyConnectedRecursiveLayer (int id, int numberOfUnits, HiddenFunction activationFunction) {
        super (id, numberOfUnits, activationFunction);
        this.loopWeights = new double[numberOfUnits][numberOfUnits];
        this.layerSize = numberOfUnits;
    }
    
    // TODO
    @Override
    public FullyConnectedLayer feedForwardCopy () {
        FullyConnectedLayer clone = new FullyConnectedLayer (id, this.getNumberOfUnits (), this.getActivationFunction ());
        clone.setBias            (this.getBias ());
        clone.setBiasErrors      (this.getBiasErrors ());
        clone.setErrWrtInnerP    (this.getErrWrtInnerP ());
        clone.setInnerPotentials (this.getInnerPotentials ());
        clone.setOutput          (this.getOutput ());
        clone.setWeightErrors    (this.getWeightErrors ());
        clone.setWeights         (this.getWeights ());
        return clone;
    }
    
    @Override
    public void backwardPass () {
        throw new UnsupportedOperationException ("Not supported! Use backpropagation on unfolded network.");
    }
    
    @Override
    public void forwardPass () {
        double[] input = this.getInputMerger().getOutput();
        
        HiddenFunction activationFunction = this.getActivationFunction ();
        double[] bias = this.getBias ();
        double[][] forwardWeights = this.getWeights ();
        double[] innerPotentials = this.getInnerPotentials ();
        double[] output = new double [layerSize];
        
        for (int n = 0; n < layerSize; n++) {
            double previousPotential = innerPotentials[n];
            innerPotentials[n] = bias[n];
            for (int i = 0; i < forwardWeights[n].length; i++) {
                innerPotentials[n] += input[i] * forwardWeights[n][i];
            }
            for (int i = 0; i < layerSize; i++) {
                innerPotentials[n] += previousPotential * loopWeights[n][i];
            }
            output[n] = activationFunction.apply (innerPotentials[n]);
        }
        
        this.setOutput (output);
    }
    
    @Override
    public void initializeWeights (long seed) {
        super.initializeWeights (seed);
        Random random = new Random (seed);
        
        // Initialize the loop weights.
        for (int i = 0; i < layerSize; i++) {
            for (int i2 = 0; i2 < layerSize; i2++) {
                double x = Math.sqrt (1.0 / layerSize);
                this.loopWeights[i][i2] = uniformRandom (random, -x, x);
            }
        }
    }
    
    private static double uniformRandom (Random random, double from, double to) {
        // Random double is in range <0, 1>.
        double randomValue = random.nextDouble ();
        return randomValue * (to - from) - from;
    }
}
