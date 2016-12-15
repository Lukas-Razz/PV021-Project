package cz.pv021.neuralnets.layers;

import java.util.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import cz.pv021.neuralnets.functions.HiddenFunction;

/**
 * A recursive layer, i.e. layer with self-loops.
 * 
 * @author  Lukáš Daubner, Josef Plch
 * @since   2016-10-30
 * @version 2016-12-15
 */
public class FullyConnectedRecursiveLayer extends FullyConnectedLayer implements RecurrentHiddenLayer {
    private final Logger logger = LoggerFactory.getLogger (FullyConnectedRecursiveLayer.class);

    public FullyConnectedRecursiveLayer (int id, int numberOfUnits, HiddenFunction activationFunction) {
        super (id, numberOfUnits, activationFunction);
    }
    
    @Override
    public void backwardPass () {
        throw new UnsupportedOperationException ("Not supported! Use backpropagation on unfolded network.");
    }
    
    /*
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
    */
    
    /*
    @Deprecated
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
    */
    
    // TODO
    @Override
    public FullyConnectedLayer makeNonRecurrentCopy (int id) {
        FullyConnectedLayer copy = new FullyConnectedLayer (id, this.getNumberOfUnits (), this.getActivationFunction ());
        copy.setBias            (this.getBias ());
        copy.setBiasErrors      (this.getBiasErrors ());
        copy.setErrWrtInnerP    (this.getErrWrtInnerP ());
        copy.setInnerPotentials (this.getInnerPotentials ());
        copy.setOutput          (this.getOutput ());
        copy.setWeightErrors    (this.getWeightErrors ());
        copy.setWeights         (this.getWeights ());
        return copy;
    }
    
    @Override
    public String toString () {
        return ("FullyConnectedRecurrentLayer {id=" + this.getId () + "}");
    }
    
    private static double uniformRandom (Random random, double from, double to) {
        // Random double is in range <0, 1>.
        double randomValue = random.nextDouble ();
        return randomValue * (to - from) - from;
    }
}
