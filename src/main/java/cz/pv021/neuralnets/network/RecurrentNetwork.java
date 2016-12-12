package cz.pv021.neuralnets.network;

import cz.pv021.neuralnets.error.Cost;
import cz.pv021.neuralnets.layers.InputLayer;
import cz.pv021.neuralnets.layers.LayerWithInput;
import cz.pv021.neuralnets.layers.Layers;
import cz.pv021.neuralnets.layers.OutputLayer;
import cz.pv021.neuralnets.layers.RecursiveHiddenLayer;
import cz.pv021.neuralnets.utils.LayerParameters;
import java.util.List;
import cz.pv021.neuralnets.optimizers.Optimizer;
import cz.pv021.neuralnets.utils.OutputExample;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * @param <IL> Type of the input layer.
 * @param <OL> Type of the output layer.
 * 
 * @author  Josef Plch
 * @since   2016-12-11
 * @version 2016-12-12
 */
public class RecurrentNetwork <IL extends InputLayer, OL extends OutputLayer> implements Network {
    private final IL inputLayer;
    private final List <RecursiveHiddenLayer> hiddenLayers;
    private final OL outputLayer;
    private final Cost cost;
    private final Optimizer optimizer;
    
    public RecurrentNetwork (IL inputLayer, List <RecursiveHiddenLayer> hiddenLayers, OL outputLayer, Cost cost, Optimizer optimizer) {
        this.inputLayer = inputLayer;
        this.hiddenLayers = hiddenLayers;
        this.outputLayer = outputLayer;
        this.cost = cost;
        this.optimizer = optimizer;
        this.outputLayer.setLoss (cost.getLoss ());
        connectLayers ();
    }
    
    private double[] zeros (int n) {
        double[] result = new double[n];
        Arrays.fill (result, 0);
        return result;
    }
    
    private void unfold (int i, int k) {
        this.hiddenLayers.get(i).unfold(k);
    }
    
    private void fold (int i, int k) {
        List <RecursiveHiddenLayer> unfolded = this.hiddenLayers.subList (i, i + k);
        Layers.connect (unfolded.get(0).getUpperLayer(), unfolded.get(0));
        Layers.connect (unfolded.get(k-1), unfolded.get(k-1).getLowerLayer());
    }
    
    // a[t] is the input at time t.
    // y[t] is the output
    private void backpropagationThroughTime (List <double[]> a, double y[]) {   
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
            this.inputLayer.setInput (x);
            this.forwardPass ();
            for (int i = t; i < t+k-1; i++) {
                this.inputLayer.setInput (a.get (i));
                this.forwardPass ();
            }
            
            // p = forward-propagate the inputs over the whole unfolded network
            // error = target - prediction
            // e = y[t+k] - p;
            this.setExpectedOutput (y[t+k]);
            
            // Back-propagate the error, e, back across the whole unfolded network.
            // Sum the weight changes in the k instances of f together.
            // Update all the weights in f and g.
            this.backwardPass ();

            // Compute the context for the next time-step.
            // x = f (x, a[t]);
        }
        
        this.fold (hiddenIndex, k);
    }
    
    private void adaptLayerWeights (LayerWithInput layer) {
        LayerParameters parameters = layer.getParameters ();
        List<LayerParameters> gradients = layer.getErrors();
        layer.setParameters (optimizer.changeParameters (parameters, gradients));
        layer.resetGradients();
    }
    
    public void adaptWeights () {
        hiddenLayers.forEach (hiddenLayer -> {
            adaptLayerWeights (hiddenLayer);
        });
        adaptLayerWeights (outputLayer);
    }
    
    public void backwardPass () {
        // The layers must be processed in reverse order.
        outputLayer.backwardPass ();
        for (int i = hiddenLayers.size () - 1; i >= 0; i--) {
            hiddenLayers.get(i).backwardPass ();
        }
    }
    
    private void connectLayers () {
        if (hiddenLayers.isEmpty ()) {
            Layers.connect (inputLayer, outputLayer);
        }
        else {
            Layers.connect (inputLayer, hiddenLayers.get (0));
            
            int numberOfHiddenLayers = hiddenLayers.size ();
            for (int i = 0; i < numberOfHiddenLayers - 1; i++) {
                Layers.connect (hiddenLayers.get (i), hiddenLayers.get (i + 1));
            }
            
            Layers.connect (hiddenLayers.get (numberOfHiddenLayers - 1), outputLayer);
        }
    }
    
    public void forwardPass () {
        hiddenLayers.forEach (hiddenLayer -> {
            hiddenLayer.forwardPass ();
        });
        outputLayer.forwardPass ();
    }
    
    public IL getInputLayer () {
        return inputLayer;
    }
    
    // Delegate method.
    public double[] getOutput () {
        return outputLayer.getOutput ();
    }
    
    // Delegate method.
    public int getOutputClassIndex () {
        return outputLayer.getOutputClassIndex ();
    }
    
    public OL getOutputLayer () {
        return outputLayer;
    }
    
    public void initializeWeights (long seed) {
        hiddenLayers.forEach (hiddenLayer -> {
            hiddenLayer.initializeWeights (seed);
        });
        outputLayer.initializeWeights (seed);
    }
    
    // Delegate method.
    public void setExpectedOutput (double expectedOutput) {
        outputLayer.setExpectedOutput (expectedOutput);
    }
    
    // Delegate method.
    public void setInput (double[] input) {
        inputLayer.setInput (input);
    }
    
    public List<LayerParameters> getParameters() {
        List<LayerParameters> parameters = new ArrayList<>();
        for(LayerWithInput layer : hiddenLayers) {
            parameters.add(layer.getParameters());
        }
        parameters.add(outputLayer.getParameters());
        return parameters;
    }
    
    public double getBatchError(List<OutputExample> batchOutput) {
        return cost.getBatchError(batchOutput, getParameters());
    }
}
