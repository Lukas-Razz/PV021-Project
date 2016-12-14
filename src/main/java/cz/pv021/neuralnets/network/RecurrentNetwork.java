package cz.pv021.neuralnets.network;

import cz.pv021.neuralnets.error.Cost;
import cz.pv021.neuralnets.layers.FullyConnectedRecursiveLayer;
import cz.pv021.neuralnets.layers.FullyConnectedLayer;
import cz.pv021.neuralnets.layers.HiddenLayer;
import cz.pv021.neuralnets.layers.InputLayer;
import cz.pv021.neuralnets.layers.LayerWithOutput;
import cz.pv021.neuralnets.layers.Layers;
import cz.pv021.neuralnets.layers.OutputLayer;
import cz.pv021.neuralnets.layers.RecursiveHiddenLayer;
import java.util.List;
import cz.pv021.neuralnets.optimizers.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Schema of unfolding:
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
 * 
 * @param <I>  Type of input objects.
 * @param <OL> Type of the output layer.
 * 
 * @author  Josef Plch
 * @since   2016-12-11
 * @version 2016-12-14
 */
public class RecurrentNetwork <I, OL extends OutputLayer> extends MultilayerPerceptron {
    public RecurrentNetwork (InputLayer <I> inputLayer, List <HiddenLayer> hiddenLayers, OL outputLayer, Cost cost, Optimizer optimizer) {
        super (inputLayer, hiddenLayers, outputLayer, cost, optimizer);
    }
    
    // TODO: Folding and unfolding is completely wrong.
    // (Má se rozkládat do šířky, ne do hloubky.)
    //
    // a[t] is the input at time t.
    // y[t] is the output
    public void backpropagationThroughTime (List <double[]> a, double y) {   
        int k = 3;
        int hiddenIndex = 0;
        
        int n = a.size ();
        int inputElementSize = a.get(0).length;
        
        // Unfold the network to contain k instances of f.
        this.unfold (hiddenIndex, k);
        
        // x = the current context
        // x0 = the zero-magnitude vector
        double[] x = zeros (inputElementSize);

        // t = time
        // n = the length of the training sequence
        for (int t = 0; t < n - k; t++) {
            // Set the network inputs to x, a[t], a[t+1], ..., a[t+k-1]
            this.getInputLayer().setInput (x);
            this.forwardPass ();
            for (int i = t; i < t+k-1; i++) {
                this.getInputLayer().setInput (a.get (i));
                this.forwardPass ();
            }
            
            // p = forward-propagate the inputs over the whole unfolded network
            // error = target - prediction
            // e = y[t+k] - p;
            this.setExpectedOutput (y);
            
            // Back-propagate the error, e, back across the whole unfolded network.
            // Sum the weight changes in the k instances of f together.
            // Update all the weights in f and g.
            this.backwardPass ();
            
            // Compute the context for the next time-step.
            // x = f (x, a[t]);
        }
        
        this.fold (hiddenIndex, k);
    }
    
    // Let k layers collapse into a single recurent one.
    private void fold (int i, int k) {
        List <HiddenLayer> hiddenLayers = this.getHiddenLayers ();
        
        // Simple sublist is just a view of the original list!
        List <HiddenLayer> beforeUnfolded = new ArrayList (hiddenLayers.subList (0, i));
        List <HiddenLayer> unfoldedLayers = new ArrayList (hiddenLayers.subList (i, i + k));
        List <HiddenLayer> afterUnfolded  = new ArrayList (hiddenLayers.subList (i + k, hiddenLayers.size ()));
        
        HiddenLayer firstUnfolded = unfoldedLayers.get (0);
        int layerSize = firstUnfolded.getNumberOfUnits ();
        FullyConnectedRecursiveLayer folded = new FullyConnectedRecursiveLayer (
            -1, //IDčko
            layerSize,
            firstUnfolded.getActivationFunction ()
        );
        
        // TODO: Update the other attributes.
        double[] innerPotentialSums = new double[layerSize];
        for (HiddenLayer unfoldedLayer : unfoldedLayers) {
            double[] innerPotentials = unfoldedLayer.getInnerPotentials ();
            for (int i2 = 0; i2 < layerSize; i2++) {
                innerPotentialSums[i2] += innerPotentials[i2];
            }
        }
        for (int i2 = 0; i2 < layerSize; i2++) {
            innerPotentialSums[i2] /= k;
        }
        folded.setInnerPotentials (innerPotentialSums);
        
        hiddenLayers.clear ();
        hiddenLayers.addAll (beforeUnfolded);
        hiddenLayers.add (folded);
        hiddenLayers.addAll (afterUnfolded);
        
        this.connectLayers ();
    }
    
    private void unfold (int i, int k) {
        List <HiddenLayer> hiddenLayers = this.getHiddenLayers ();
        HiddenLayer layer = hiddenLayers.get (i);
        if (! (layer instanceof RecursiveHiddenLayer)) {
            throw new IllegalArgumentException ("Trying to unfold non-recurrent layer.");
        }
        
        hiddenLayers.remove (i);
        RecursiveHiddenLayer recurrentLayer = (RecursiveHiddenLayer) layer;
        int layerSize = recurrentLayer.getNumberOfUnits ();
        FullyConnectedLayer zeroContextLayer = new FullyConnectedLayer (-1, layerSize, null); //IDčko
        zeroContextLayer.setOutput (zeros (layerSize));
        
        List <LayerWithOutput> inputForNextLayer = new ArrayList <> ();
        inputForNextLayer.addAll (recurrentLayer.getInputLayers ());
        inputForNextLayer.add (zeroContextLayer);
        
        // TODO.
        for (int t = 0; t < k; t++) {
            HiddenLayer unfoldedLayer = recurrentLayer.feedForwardCopy ();
            Layers.connect (inputForNextLayer, unfoldedLayer);
            hiddenLayers.add (i + t, unfoldedLayer);
            
            inputForNextLayer = new ArrayList <> ();
            inputForNextLayer.add (unfoldedLayer);
            inputForNextLayer.add (zeroContextLayer);
        }
        
        this.connectLayers ();
    }
    
    private static double[] zeros (int n) {
        double[] result = new double[n];
        Arrays.fill (result, 0);
        return result;
    }
}
