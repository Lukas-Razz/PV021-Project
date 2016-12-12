package cz.pv021.neuralnets.layers;

import cz.pv021.neuralnets.functions.ActivationFunction;
import java.util.Arrays;
import java.util.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A recursive layer, i.e. layer with self-loops.
 * 
 * TODO: Rename (there is a missing T in the class name).
 * 
 * @author  Lukáš Daubner, Josef Plch
 * @since   2016-10-30
 * @version 2016-12-12
 */
public class FullyConnecedRecursiveLayer extends FullyConnectedLayer implements RecursiveHiddenLayer {
    private final Logger logger = LoggerFactory.getLogger (FullyConnecedRecursiveLayer.class);
    private double[][] loopWeights;
    private final int layerSize;

    public FullyConnecedRecursiveLayer (int numberOfUnits, ActivationFunction activationFunction) {
        super (numberOfUnits, activationFunction);
        this.loopWeights = new double[numberOfUnits][numberOfUnits];
        this.layerSize = numberOfUnits;
    }
    
    /**
     * Unfolding schema:
     * 
     *    y                        y_t-1     y_t       y_t+1
     *    O                        O         O         O
     *    ↑                        ↑         ↑         ↑
     *  V | ___                  V |       V |       V |
     *    |/   \     unfold        | s_t-1   | s_t     | s_t+1
     *  s O     | W  =====> ------>O-------->O-------->O-------->
     *    ↑\___/              W    ↑    W    ↑    W    ↑    W
     *  U |                      U |       U |       U |
     *    |                        |         |         |
     *    O                        O         O         O
     *    x                        x_t-1     x_t       x_t+1
     * 
     * @param k Number of layers to unfold to.
     */
    @Override
    public void unfold (int k) {
        FullyConnectedLayer zeroContextLayer = new FullyConnectedLayer (layerSize, null);
        zeroContextLayer.setOutput (zeros (layerSize));
        
        InputMergeLayer inputLayer = new InputMergeLayer (this.getUpperLayer (), zeroContextLayer);
        for (int i = 0; i < k; i++) {
            FullyConnectedLayer ffCopy = this.feedForwardCopy ();
            Layers.connect (inputLayer, ffCopy);
            inputLayer = new InputMergeLayer (ffCopy, inputLayer);
        }
        Layers.connect (inputLayer, this.getLowerLayer ());
    }

    
    @Override
    public FullyConnectedLayer feedForwardCopy () {
        FullyConnectedLayer clone = new FullyConnectedLayer (this.getNumberOfUnits (), this.getActivationFunction ());
        return clone;
    }
    
    @Override
    public void backwardPass () {
        throw new UnsupportedOperationException ("Not supported! Use backpropagation on unfolded network.");
    }
    
    @Override
    public void forwardPass () {
        double[] input = this.getUpperLayer().getOutput ();
        
        ActivationFunction activationFunction = this.getActivationFunction ();
        double[] bias = this.getBias ();
        double[][] ffWeights = this.getWeights ();
        double[] innerPotentials = this.getInnerPotentials ();
        double[] output = new double [layerSize];
        
        for (int n = 0; n < layerSize; n++) {
            double previousPotential = innerPotentials[n];
            innerPotentials[n] = bias[n];
            for (int i = 0; i < ffWeights[n].length; i++) {
                innerPotentials[n] += input[i] * ffWeights[n][i];
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
    
    private double uniformRandom (Random random, double from, double to) {
        double randomValue = random.nextDouble ();
        return randomValue * (to - from) - from;
    }
    
    private double[] zeros (int n) {
        double[] result = new double[n];
        Arrays.fill (result, 0);
        return result;
    }
}
