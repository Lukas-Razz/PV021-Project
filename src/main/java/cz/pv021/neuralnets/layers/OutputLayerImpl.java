package cz.pv021.neuralnets.layers;

import cz.pv021.neuralnets.error.Cost;
import cz.pv021.neuralnets.error.Loss;
import cz.pv021.neuralnets.functions.OutputFunction;
import cz.pv021.neuralnets.utils.LayerParameters;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Implementation of output layer.
 * 
 * @author  Lukáš Daubner
 * @since   2016-10-30
 * @version 2016-12-06
 */
public class OutputLayerImpl implements OutputLayer {
    final Logger logger = LoggerFactory.getLogger(OutputLayerImpl.class);
    
    private Loss loss;
    private final OutputFunction outputFunction;
    private final int numberOfUnits;
    private Layer upperLayer; // Vstupni
    private double[][] weights;
    private double[] bias;
    
    private double[] innerPotentials;
    private double[] err_wrt_innerP;
    
    private List<double[][]> weightErrors;
    private List<double[]> biasErrors;
       
    private double[] output;
    private double expectedOutput;

    public OutputLayerImpl (int numberOfUnits, OutputFunction outputFunction, Loss loss) {
        this.numberOfUnits = numberOfUnits;
        this.output = new double[numberOfUnits];
        this.bias = new double[numberOfUnits];
        this.outputFunction = outputFunction;
        this.loss = loss;
        
        this.innerPotentials = new double[numberOfUnits];
        this.err_wrt_innerP = new double[numberOfUnits];
        
        weightErrors = new ArrayList<>();
        biasErrors = new ArrayList<>();
        
    }
    
    @Override
    public void backwardPass () {
        double[][] err_wrt_weight = new double[numberOfUnits][upperLayer.getNumberOfUnits()];
        double[] preSoftmax = outputFunction.derivative(innerPotentials);
        
        for(int i=0; i<numberOfUnits; i++) {
            double error = i == expectedOutput ? loss.derivative(output[i], 1) : loss.derivative(output[i], 0); //pro klasifikaci
            err_wrt_innerP[i] = error * preSoftmax[i];
            
            for(int j=0; j<upperLayer.getNumberOfUnits(); j++) {
                err_wrt_weight[i][j] = err_wrt_innerP[i] * upperLayer.getOutput()[j]; // innerPotential of neuron "i" * output of neuron "j"
            }
        }
        biasErrors.add(err_wrt_innerP); // err_wrt_innerP = err_wrt_bias
        weightErrors.add(err_wrt_weight);
    }
    
    @Override
    //Inner potencial is remembered for backward pass
    public void forwardPass () {
        double[] input = upperLayer.getOutput ();
        
        for (int n = 0; n < numberOfUnits; n++) {
            innerPotentials[n] = bias[n];
            for (int i = 0; i < weights[n].length; i++) {
                innerPotentials[n] += input[i] * weights[n][i];
            }
        }
        this.output = outputFunction.apply (innerPotentials);
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
    public void initializeWeights (long seed) {
        Random r = new Random (seed);
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = Math.abs(r.nextGaussian ()); //Softmax nerad minus
            }
        }
        for (int i = 0; i < bias.length; i++) {
            bias[i] = 0;
        }
    }
    
    @Override
    public void setUpperLayer (LayerWithOutput layer) {
        this.upperLayer = layer;
        this.weights = new double[numberOfUnits][layer.getNumberOfUnits ()];
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
    public void setExpectedOutput(double expectedOutput) {
        this.expectedOutput = expectedOutput;
    }
}
