package cz.pv021.neuralnets.network;

import cz.pv021.neuralnets.error.Cost;
import cz.pv021.neuralnets.initialization.Initializer;
import cz.pv021.neuralnets.layers.HiddenLayer;
import cz.pv021.neuralnets.layers.InputLayer;
import cz.pv021.neuralnets.layers.LayerWithInput;
import cz.pv021.neuralnets.layers.LayerWithOutput;
import cz.pv021.neuralnets.layers.Layers;
import cz.pv021.neuralnets.layers.OutputLayer;
import cz.pv021.neuralnets.utils.LayerParameters;
import java.util.List;
import cz.pv021.neuralnets.optimizers.Optimizer;
import cz.pv021.neuralnets.utils.OutputExample;
import java.util.ArrayList;

/**
 * @param <IL> Type of the input layers.
 * @param <OL> Type of the output layer.
 * 
 * @author  Lukáš Daubner, Josef Plch
 * @since   2016-11-17
 * @version 2016-12-16
 */
public class MultilayerPerceptron <IL extends InputLayer, OL extends OutputLayer> implements Network {
    private final List <IL> inputLayers = new ArrayList <> ();
    private final List <HiddenLayer> hiddenLayers = new ArrayList <> ();
    private final OL outputLayer;
    private final Cost cost;
    private final Optimizer optimizer;
    
    public MultilayerPerceptron (List <IL> inputLayers, List <HiddenLayer> hiddenLayers, OL outputLayer, Cost cost, Optimizer optimizer) {
        this.inputLayers.addAll (inputLayers);
        this.hiddenLayers.addAll (hiddenLayers);
        this.outputLayer = outputLayer;
        this.cost = cost;
        this.optimizer = optimizer;
        this.outputLayer.setLoss (cost.getLoss ());
        connectLayers ();
    }
    
    private void adaptLayerWeights (LayerWithInput layer) {
        LayerParameters parameters = layer.getParameters ();
        List<LayerParameters> gradients = layer.getErrors ();
        layer.setParameters (optimizer.changeParameters (parameters, gradients));
        layer.resetGradients ();
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
    
    public final void connectLayers () {
        List <LayerWithOutput> ils = new ArrayList <> (inputLayers);
        
        if (hiddenLayers.isEmpty ()) {
            Layers.connect (ils, outputLayer);
        }
        else {
            Layers.connect (ils, hiddenLayers.get (0));
            
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
    
    public double getBatchError (List <OutputExample> batchOutput) {
        return cost.getBatchError (batchOutput, getParameters ());
    }
    
    public List <HiddenLayer> getHiddenLayers () {
        return hiddenLayers;
    }
    
    public List <IL> getInputLayers () {
        return inputLayers;
    }
    
    @Override
    public int getNumberOfUnits () {
        int sum = 0;
        for (IL inputLayer : inputLayers) {
            sum += inputLayer.getNumberOfUnits ();
        }
        for (HiddenLayer hiddenLayer : hiddenLayers) {
            sum += hiddenLayer.getNumberOfUnits ();
        }
        sum += outputLayer.getNumberOfUnits ();
        return sum;
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
    
    public List <LayerParameters> getParameters () {
        List <LayerParameters> parameters = new ArrayList <> ();
        for (LayerWithInput layer : hiddenLayers) {
            parameters.add(layer.getParameters ());
        }
        parameters.add (outputLayer.getParameters ());
        return parameters;
    }
    
    @Override
    public void initializeWeights (Initializer initializer) {
        hiddenLayers.forEach (hiddenLayer -> {
            hiddenLayer.resetWeights ();
            initializer.initialize (hiddenLayer.getParameters (), hiddenLayer.getActivationFunction ());
        });
        outputLayer.resetWeights ();
        initializer.initialize(outputLayer.getParameters(), outputLayer.getActivationFunction());
    }
    
    // Delegate method.
    public void setExpectedOutput (double expectedOutput) {
        outputLayer.setExpectedOutput (expectedOutput);
    }
    
    // Delegate method.
    public void setInputs (List <double[]> inputs) {
        for (int i = 0; i < inputs.size (); i++) {
            inputLayers.get (i).setInput (inputs.get (i));
        }
    }
    
    @Override
    public String toString () {
        return (
            "MultilayerPerceptron (" + this.getNumberOfUnits() + " neurons) {"
                + "\n    inputLayers=" + inputLayers
                + ",\n    hiddenLayers=" + hiddenLayers
                + ",\n    outputLayer=" + outputLayer
            + "\n}"
        );
    }
}
