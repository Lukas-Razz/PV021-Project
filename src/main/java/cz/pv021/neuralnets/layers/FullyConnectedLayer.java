package cz.pv021.neuralnets.layers;

import cz.pv021.neuralnets.utils.LayerParameters;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import cz.pv021.neuralnets.functions.HiddenFunction;

/**
 * @author  Lukáš Daubner
 * @since   2016-10-30
 * @version 2016-12-15
 */
public class FullyConnectedLayer implements HiddenLayer {
    private final Logger logger = LoggerFactory.getLogger(FullyConnectedLayer.class);
    private final int id;
    private final HiddenFunction activationFunction;
    private InputMerger inputMerger;
    private LayerWithInput outputLayer;
    private double[] bias;
    private List<double[]> biasErrors;
    private double[] err_wrt_innerP; // Error with respect to inner potential.
    private double[] innerPotentials;
    private final int numberOfUnits;
    private double[] output;
    private List<double[][]> weightErrors;
    private double[][] weights;

    public FullyConnectedLayer (int id, int numberOfUnits, HiddenFunction activationFunction) {
        this.id = id;
        this.numberOfUnits = numberOfUnits;
        this.output = new double[numberOfUnits];
        this.bias = new double[numberOfUnits];
        this.activationFunction = activationFunction;
        
        this.innerPotentials = new double[numberOfUnits];
        this.err_wrt_innerP = new double[numberOfUnits];
        
        weightErrors = new ArrayList<>();
        biasErrors = new ArrayList<>();
    }

    @Override
    public void backwardPass () {
        String logPrefix = "FCL #" + id + " / backwardPass: ";
        // System.out.println (logPrefix + "innerPotentials = " + Arrays.toString (innerPotentials));
        
        // Error with respect to weight.
        double[][] e_wrt_weight = new double[numberOfUnits][inputMerger.getNumberOfUnits()];
        
        for (int i=0; i<numberOfUnits; i++) {
            err_wrt_innerP[i] = 0;
            for (int k=0; k<outputLayer.getNumberOfUnits(); k++) {
                 err_wrt_innerP[i] += outputLayer.getInnerPotentialGradient()[k] * outputLayer.getParameters().getWeights()[k][i];
            }
            err_wrt_innerP[i] = err_wrt_innerP[i] * activationFunction.derivative(innerPotentials[i]);
            
            for (int j=0; j<inputMerger.getNumberOfUnits(); j++) {
                // innerPotential of neuron "i" * output of neuron "j"
                e_wrt_weight[i][j] = err_wrt_innerP[i] * inputMerger.getOutput()[j];
            }
        }
        // System.out.println (logPrefix + "gradient = " + Arrays.toString (this.getInnerPotentialGradient ()));
        biasErrors.add(err_wrt_innerP); // e_wrt_innerP = e_wrt_bias
        weightErrors.add(e_wrt_weight);
    }

    @Override
    public void forwardPass () {
        String logPrefix = "FCL #" + id + " / forwardPass: ";
        double[] input = inputMerger.getOutput ();
        // System.out.println (logPrefix + "input = " + Arrays.toString (input));
        for (int n = 0; n < numberOfUnits; n++) {
            // System.out.println (logPrefix + "weights[" + n + "] = " + Arrays.toString (weights[n]));
            innerPotentials[n] = bias[n];
            for (int i = 0; i < weights[n].length; i++) {
                innerPotentials[n] += input[i] * weights[n][i];
            }
            output[n] = activationFunction.apply (innerPotentials[n]);
        }
    }
    
    @Override
    public HiddenFunction getActivationFunction () {
        return activationFunction;
    }
    
    public double[] getBias () {
        return bias;
    }
    
    public List <double[]> getBiasErrors () {
        return biasErrors;
    }
    
    @Override
    public List<LayerParameters> getErrors () {
        List<LayerParameters> errors = new ArrayList <> ();
        for (int i = 0; i < weightErrors.size (); i++) {
            errors.add (new LayerParameters (weightErrors.get(i), biasErrors.get(i), id));
        }
        return errors;
    }

    public double[] getErrWrtInnerP () {
        return err_wrt_innerP;
    }
    
    @Override
    public int getId () {
        return id;
    }
    
    // Alias.
    @Override
    public double[] getInnerPotentialGradient () {
        return this.getErrWrtInnerP ();
    }

    @Override
    public double[] getInnerPotentials () {
        return innerPotentials;
    }
    
    @Override
    public List <LayerWithOutput> getInputLayers () {
        return inputMerger.getLayers ();
    }
    
    @Override
    public int getInputSize () {
        return inputMerger.getNumberOfUnits ();
    }
    
    /*
    public InputMerger getInputMerger () {
        return inputMerger;
    }
    */
    
    @Override
    public int getNumberOfUnits () {
        return numberOfUnits;
    }

    @Override
    public double[] getOutput () {
        return output;
    }
    
    @Override
    public LayerWithInput getOutputLayer () {
        return outputLayer;
    }
    
    @Override
    public LayerParameters getParameters () {
        return new LayerParameters (weights, bias, id);
    }
    
    public List <double[][]> getWeightErrors () {
        return weightErrors;
    }
    
    @Override
    public double[][] getWeights () {
        return weights;
    }
    
    @Deprecated
    @Override
    public void initializeWeights (long seed) {
        System.out.println ("Initializing weights for layer #" + id);
        this.resetWeights ();
        
        Random r = new Random (seed);
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = r.nextGaussian ();
            }
        }
        for (int i = 0; i < bias.length; i++) {
            bias[i] = 0;
        }
    }
    
    @Override
    public void resetGradients () {
        biasErrors.clear ();
        weightErrors.clear ();
    }
    
    @Override
    public void resetWeights () {
        System.out.println ("RESETING WEIGHTS: " + inputMerger.getLayers().size() + " input layers, " + inputMerger.getNumberOfUnits () + " input neurons in total");
        this.weights = new double[numberOfUnits][inputMerger.getNumberOfUnits ()];
    }
    
    public void setBias (double[] bias) {
        this.bias = bias;
    }
    
    public void setBiasErrors (List <double[]> biasErrors) {
        this.biasErrors = biasErrors;
    }
    
    public void setErrWrtInnerP (double[] errWrtInnerP) {
        this.err_wrt_innerP = errWrtInnerP;
    }
    
    // Alias.
    public void setInnerPotentialGradient (double[] innerPotentialGradient) {
        this.setErrWrtInnerP (innerPotentialGradient);
    }
    
    public void setInnerPotentials (double[] innerPotentials) {
        this.innerPotentials = innerPotentials;
    }
    
    @Override
    public void setInputLayers (List <LayerWithOutput> layers) {
        this.inputMerger = new InputMerger (layers);
    }

    public void setOutput (double[] output) {
        this.output = output;
    }
    
    @Override
    public void setOutputLayer (LayerWithInput outputLayer) {
        this.outputLayer = outputLayer;
    }
    
    @Override
    public void setParameters (LayerParameters parameters) {
        this.bias = parameters.getBias ();
        this.weights = parameters.getWeights ();
    }
    
    public void setWeightErrors (List <double[][]> weightErrors) {
        this.weightErrors = weightErrors;
    }
    
    public void setWeights (double[][] weights) {
        this.weights = weights;
    }
    
    @Override
    public String toString () {
        List <Integer> inputLayerIds = new ArrayList <> ();
        for (LayerWithOutput inputLayer : this.getInputLayers ()) {
            inputLayerIds.add (inputLayer.getId ());
        }
        return (
            "FullyConnectedLayer {"
                + "id=" + id
                + ", inputLayers=" + inputLayerIds
                + ", outputLayer=" + outputLayer.getId ()
            + "}"
        );
    }
}
