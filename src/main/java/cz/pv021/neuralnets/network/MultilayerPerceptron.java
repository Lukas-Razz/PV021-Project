package cz.pv021.neuralnets.network;

import cz.pv021.neuralnets.layers.HiddenLayer;
import cz.pv021.neuralnets.layers.InputLayer;
import cz.pv021.neuralnets.layers.LayerWithInput;
import cz.pv021.neuralnets.layers.Layers;
import cz.pv021.neuralnets.layers.OutputLayer;
import cz.pv021.neuralnets.optimalizers.SGD;
import cz.pv021.neuralnets.utils.LayerParameters;
import java.util.List;

/**
 * @param <IL> Type of the input layer.
 * @param <OL> Type of the output layer.
 * 
 * @author  Lukáš Daubner, Josef Plch
 * @since   2016-11-17
 * @version 2016-12-04
 */
public class MultilayerPerceptron <IL extends InputLayer, OL extends OutputLayer> implements Network {
    private final IL inputLayer;
    private final List <HiddenLayer> hiddenLayers;
    private final OL outputLayer;
    
    public MultilayerPerceptron (IL inputLayer, List <HiddenLayer> hiddenLayers, OL outputLayer) {
        this.inputLayer = inputLayer;
        this.hiddenLayers = hiddenLayers;
        this.outputLayer = outputLayer;
        connectLayers ();
    }
    
    private void adaptLayerWeights (SGD adapter, LayerWithInput layer) {
        LayerParameters parameters = layer.getParameters ();
        LayerParameters gradients = layer.getErrors().get(0);
        // TODO: Tady bylo (principialne):
        // outputLayer.setParameters (...)
        // Predpokladam, ze to byl preklep, ale radsi zkontrolovat.
        layer.setParameters (adapter.changeParameters (parameters, gradients));
    }
    
    public void adaptWeights (SGD adapter) {
        hiddenLayers.forEach (hiddenLayer -> {
            adaptLayerWeights (adapter, hiddenLayer);
        });
        adaptLayerWeights (adapter, outputLayer);
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
    public void setInput (double[] input) {
        inputLayer.setInput (input);
    }
    
    // Delegate method.
    public void setExpectedOutput (double expectedOutput) {
        outputLayer.setExpectedOutput (expectedOutput);
    }
}
