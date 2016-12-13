package cz.pv021.neuralnets.layers;

import cz.pv021.neuralnets.functions.ActivationFunction;
import cz.pv021.neuralnets.utils.LayerParameters;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author  Lukáš Daubner
 * @since   2016-10-30
 * @version 2016-12-07
 */
public class FullyConnectedLayer implements HiddenLayer {
    private final Logger logger = LoggerFactory.getLogger(FullyConnectedLayer.class);
    
    private final ActivationFunction activationFunction;
    private LayerWithInput outputLayer;
    private LayerWithOutput inputLayer;
    private final int numberOfUnits;
    private double[] output;
    private double[][] weights;
    private double[] bias;
    
    private double[] innerPotentials;
    private double[] err_wrt_innerP; // Error with respect to inner potential.
    
    private List<double[][]> weightErrors;
    private List<double[]> biasErrors;

    public FullyConnectedLayer (int numberOfUnits, ActivationFunction activationFunction) {
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
        // Error with respect to weight.
        double[][] e_wrt_weight = new double[numberOfUnits][inputLayer.getNumberOfUnits()];
        
        for (int i=0; i<numberOfUnits; i++) {
            err_wrt_innerP[i] = 0;
            for (int k=0; k<outputLayer.getNumberOfUnits(); k++) {
                 err_wrt_innerP[i] += outputLayer.getInnerPotentialGradient()[k] * outputLayer.getParameters().getWeights()[k][i];
            }
            err_wrt_innerP[i] = err_wrt_innerP[i] * activationFunction.derivative(innerPotentials[i]);
            
            for (int j=0; j<inputLayer.getNumberOfUnits(); j++) {
                // innerPotential of neuron "i" * output of neuron "j"
                e_wrt_weight[i][j] = err_wrt_innerP[i] * inputLayer.getOutput()[j];
            }
        }
        biasErrors.add(err_wrt_innerP); // e_wrt_innerP = e_wrt_bias
        weightErrors.add(e_wrt_weight);
    }

    @Override
    public void forwardPass () {
        double[] input = inputLayer.getOutput ();
        for (int n = 0; n < numberOfUnits; n++) {
            innerPotentials[n] = bias[n];
            for (int i = 0; i < weights[n].length; i++) {
                innerPotentials[n] += input[i] * weights[n][i];
            }
            output[n] = activationFunction.apply (innerPotentials[n]);
        }
    }
    
    @Override
    public LayerWithInput getOutputLayer () {
        return outputLayer;
    }
    
    @Override
    public int getNumberOfUnits () {
        return numberOfUnits;
    }

    @Override
    public double[] getOutput () {
        return output;
    }
    
    @Override
    public LayerWithOutput getInputLayer () {
        return inputLayer;
    }
    
    @Override
    public void initializeWeights (long seed) {
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
    public void setOutputLayer (LayerWithInput outputLayer) {
        this.outputLayer = outputLayer;
    }
    
    @Override
    public void setInputLayer (LayerWithOutput inputLayer) {
        this.inputLayer = inputLayer;
        weights = new double[numberOfUnits][inputLayer.getNumberOfUnits ()];
    }

    @Override
    public LayerParameters getParameters() {
        return new LayerParameters(weights, bias);
    }
    
    @Override
    public void setParameters(LayerParameters parameters) {
        weights = parameters.getWeights();
        bias = parameters.getBias();
    }

    @Override
    public double[] getInnerPotentialGradient() {
        return err_wrt_innerP;
    }

    @Override
    public List<LayerParameters> getErrors() {
        List<LayerParameters> errors = new ArrayList<>();
        for(int i=0; i<weightErrors.size(); i++) {
            errors.add(new LayerParameters(weightErrors.get(i), biasErrors.get(i)));
        }
        return errors;
    }

    @Override
    public void resetGradients() {
        biasErrors.clear();
        weightErrors.clear();
    }
    
    @Override
    public ActivationFunction getActivationFunction () {
        return activationFunction;
    }
    
    public double[] getBias () {
        return bias;
    }
    
    public List <double[]> getBiasErrors () {
        return biasErrors;
    }
    
    public double[] getErrWrtInnerP () {
        return err_wrt_innerP;
    }
    
    public double[] getInnerPotentials () {
        return innerPotentials;
    }
    
    public List <double[][]> getWeightErrors () {
        return weightErrors;
    }
    
    public double[][] getWeights () {
        return weights;
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
    
    public void setInnerPotentials (double[] innerPotentials) {
        this.innerPotentials = innerPotentials;
    }
    
    public void setOutput (double[] output) {
        this.output = output;
    }
    
    public void setWeightErrors (List <double[][]> weightErrors) {
        this.weightErrors = weightErrors;
    }
    
    public void setWeights (double[][] weights) {
        this.weights = weights;
    }
}
