package cz.pv021.neuralnets.network;

import cz.pv021.neuralnets.error.Cost;
import cz.pv021.neuralnets.layers.HiddenLayer;
import cz.pv021.neuralnets.layers.InputLayer;
import cz.pv021.neuralnets.layers.LayerWithInput;
import cz.pv021.neuralnets.layers.Layers;
import cz.pv021.neuralnets.layers.OutputLayer;
import cz.pv021.neuralnets.utils.LayerParameters;
import java.util.List;
import cz.pv021.neuralnets.optimizers.Optimizer;

/**
 * @param <IL> Type of the input layer.
 * @param <OL> Type of the output layer.
 * 
 * @author  Lukáš Daubner, Josef Plch
 * @since   2016-11-17
 * @version 2016-12-08
 */
public class MultilayerPerceptron <IL extends InputLayer, OL extends OutputLayer> implements Network {
    private final IL inputLayer;
    private final List <HiddenLayer> hiddenLayers;
    private final OL outputLayer;
    private final Cost cost;
    private final Optimizer optimizer;
    
    public MultilayerPerceptron (IL inputLayer, List <HiddenLayer> hiddenLayers, OL outputLayer, Cost cost, Optimizer optimizer) {
        this.inputLayer = inputLayer;
        this.hiddenLayers = hiddenLayers;
        this.outputLayer = outputLayer;
        this.cost = cost;
        this.optimizer = optimizer;
        this.outputLayer.setLoss (cost.getLoss ());
        connectLayers ();
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
}
