package cz.pv021.neuralnets.network;

import cz.pv021.neuralnets.error.Cost;
import cz.pv021.neuralnets.layers.FullyConnectedLayer;
import cz.pv021.neuralnets.layers.HiddenLayer;
import cz.pv021.neuralnets.layers.InputLayer;
import cz.pv021.neuralnets.layers.LayerWithOutput;
import cz.pv021.neuralnets.layers.Layers;
import cz.pv021.neuralnets.layers.OutputLayer;
import java.util.List;
import cz.pv021.neuralnets.optimizers.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import cz.pv021.neuralnets.layers.RecurrentHiddenLayer;

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
 * @param <IL> Type of the input layers.
 * @param <OL> Type of the output layer.
 * 
 * @author  Josef Plch
 * @since   2016-12-11
 * @version 2016-12-14
 */
public class RecurrentNetwork <IL extends InputLayer, OL extends OutputLayer> extends MultilayerPerceptron <IL, OL> {
    private static final int INPUT_LAYER_ID_SHIFT = 1000;
    private static final int HIDDEN_LAYER_ID_SHIFT = 2000;
    private final IL originalInputLayer;
    private final HiddenLayer originalHiddenLayer;
    
    public RecurrentNetwork (IL inputLayer, HiddenLayer hiddenLayer, OL outputLayer, Cost cost, Optimizer optimizer) {
        super (
            Collections.singletonList (inputLayer),
            Collections.singletonList (hiddenLayer),
            outputLayer,
            cost,
            optimizer
        );
        this.originalInputLayer = inputLayer;
        this.originalHiddenLayer = hiddenLayer;
        
        hiddenLayer.setInputLayers (
            Arrays.asList (hiddenLayer, inputLayer)
        );
        hiddenLayer.resetWeights ();
    }
    
    // a[t] is the input at time t.
    // y[t] is the output
    public void backpropagationThroughTime (List <double[]> inputSequence, double expectedOutput, int k) {   
        int sequenceLength = inputSequence.size ();
        
        // System.out.println ("BPTT start: " + this);
        
        // Unfold the network to contain k instances of f.
        this.unfold (k);
        
        // System.out.println ("BPTT unfolded: " + this);
        
        // t = time
        // n = the length of the training sequence
        for (int t = 0; t < sequenceLength - k; t++) {
            // Set the network inputs to x, a[t], a[t+1], ..., a[t+k-1]
            List <IL> inputLayers = this.getInputLayers ();
            List <HiddenLayer> hiddenLayers = this.getHiddenLayers ();
            for (int i = 0; i < k; i++) {
                IL inputLayer = inputLayers.get (i);
                inputLayer.setInput (inputSequence.get (t + i));
                hiddenLayers.get(i).forwardPass ();
            }
            this.getOutputLayer().forwardPass ();
            
            // p = forward-propagate the inputs over the whole unfolded network
            // error = target - prediction
            // e = y[t+k] - p;
            this.setExpectedOutput (expectedOutput);
            
            // Back-propagate the error, e, back across the whole unfolded network.
            // Sum the weight changes in the k instances of f together.
            // Update all the weights in f and g.
            this.backwardPass ();
            
            // TODO.
            // Compute the context for the next time-step.
            // x = f (x, a[t]);
        }
        
        this.fold (k);
        
        // System.out.println ("BPTT folded: " + this);
    }
    
    // Let k layers collapse into a single recurent one.
    // See the class documentation.
    public void fold (int k) {
        int i = 0;
        
        // Restore the original inputLayer.
        List <IL> inputLayers = this.getInputLayers ();
        inputLayers.clear ();
        inputLayers.add (originalInputLayer);
        
        // Fold the hidden layers.
        List <HiddenLayer> hiddenLayers = this.getHiddenLayers ();
        
        // Simple sublist is just a view of the original list!
        List <HiddenLayer> beforeUnfolded = new ArrayList (hiddenLayers.subList (0, i));
        List <HiddenLayer> unfoldedLayers = new ArrayList (hiddenLayers.subList (i, i + k));
        List <HiddenLayer> afterUnfolded  = new ArrayList (hiddenLayers.subList (i + k, hiddenLayers.size ()));
        
        int layerSize = originalHiddenLayer.getNumberOfUnits ();
        int inputSize = originalHiddenLayer.getInputSize ();
        System.out.println ("Size (layer x input): " + layerSize + " x " + inputSize);
        
        // TODO: Update the other attributes.
        double[] avgInnerPotentials = new double[layerSize];
        double[] avgInnerPotentialGradients = new double[layerSize];
        double[][] avgWeights = new double[layerSize][inputSize];
        for (HiddenLayer unfoldedLayer : unfoldedLayers) {
            double[] innerPotentials = unfoldedLayer.getInnerPotentials ();
            double[] innerPotentialGradient = unfoldedLayer.getInnerPotentialGradient ();
            double[][] weights = unfoldedLayer.getWeights ();
            for (int i2 = 0; i2 < layerSize; i2++) {
                avgInnerPotentials[i2]         += innerPotentials[i2] / k;
                avgInnerPotentialGradients[i2] += innerPotentialGradient[i2] / k;
                for (int i3 = 0; i3 < inputSize; i3++) {
                    avgWeights[i2][i3] += weights[i2][i3] / k;
                }
            }
        }
        originalHiddenLayer.setInnerPotentials (avgInnerPotentials);
        originalHiddenLayer.setInnerPotentialGradient (avgInnerPotentialGradients);
        originalHiddenLayer.setWeights (avgWeights);
        
        hiddenLayers.clear ();
        hiddenLayers.addAll (beforeUnfolded);
        hiddenLayers.add    (originalHiddenLayer);
        hiddenLayers.addAll (afterUnfolded);
        
        // super.connectLayers ();
    }
    
    // See the class documentation.
    // Also known as: unroll.
    public void unfold (int k) {
        int i = 0;
        
        List <HiddenLayer> hiddenLayers = this.getHiddenLayers ();
        HiddenLayer layer = hiddenLayers.get (i);
        if (! (layer instanceof RecurrentHiddenLayer)) {
            // throw new IllegalArgumentException ("Trying to unfold non-recurrent layer.");
        }
        
        // Make k copies of the input layer.
        List <IL> inputLayers = this.getInputLayers ();
        inputLayers.clear ();
        for (int t = 0; t < k; t++) {
            inputLayers.add (
                (IL) originalInputLayer.makeCopy (
                    originalInputLayer.getId () + INPUT_LAYER_ID_SHIFT + t
                )
            );
        }
        
        hiddenLayers.remove (i);
        HiddenLayer recurrentLayer = layer;
        
        int layerSize = recurrentLayer.getNumberOfUnits ();
        FullyConnectedLayer zeroContextLayer = new FullyConnectedLayer (-1, layerSize, null); //IDčko
        zeroContextLayer.setOutput (zeros (layerSize));
        
        HiddenLayer previousHiddenLayer = zeroContextLayer;
        for (int t = 0; t < k; t++) {
            List <LayerWithOutput> unfoldedLayerInputs = new LinkedList <> ();
            unfoldedLayerInputs.add (previousHiddenLayer);
            unfoldedLayerInputs.add (inputLayers.get (t));
            
            HiddenLayer unfoldedLayer = recurrentLayer.makeCopy (recurrentLayer.getId () + HIDDEN_LAYER_ID_SHIFT + t);
            Layers.connect (unfoldedLayerInputs, unfoldedLayer);
            hiddenLayers.add (i + t, unfoldedLayer);
            
            previousHiddenLayer = unfoldedLayer;
        }
        
        // Connect the last unfolded layer to the output layer.
        Layers.connect (previousHiddenLayer, this.getOutputLayer ());
    }
    
    private static double[] zeros (int n) {
        double[] result = new double[n];
        Arrays.fill (result, 0);
        return result;
    }
}
