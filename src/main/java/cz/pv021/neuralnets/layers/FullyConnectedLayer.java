package cz.pv021.neuralnets.layers;

import cz.pv021.neuralnets.error.Loss;
import cz.pv021.neuralnets.functions.ActivationFunction;
import cz.pv021.neuralnets.utils.LayerParameters;
import java.util.ArrayList;
import java.util.Arrays;
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
    final Logger logger = LoggerFactory.getLogger(FullyConnectedLayer.class);
    
    private final ActivationFunction activationFunction;
    private LayerWithInput lowerLayer; // Vystupni
    private LayerWithOutput upperLayer; // Vstupni
    private final int numberOfUnits;
    private final double[] output;
    private double[][] weights;
    private double[] bias;
    
    private double[] innerPotencials;
    private double[] err_wrt_innerP;
    
    private List<double[][]> weightErrors;
    private List<double[]> biasErrors;

    public FullyConnectedLayer (int numberOfUnits, ActivationFunction activationFunction) {
        this.numberOfUnits = numberOfUnits;
        this.output = new double[numberOfUnits];
        this.bias = new double[numberOfUnits];
        this.activationFunction = activationFunction;
        
        this.innerPotencials = new double[numberOfUnits];
        this.err_wrt_innerP = new double[numberOfUnits];
        
        weightErrors = new ArrayList<>();
        biasErrors = new ArrayList<>();
    }

    @Override
    public void backwardPass () {
        double[][] e_wrt_weight = new double[numberOfUnits][upperLayer.getNumberOfUnits()];
        
        for(int i=0; i<numberOfUnits; i++) {
            err_wrt_innerP[i] = 0;
            for (int k=0; k<lowerLayer.getNumberOfUnits(); k++) {
                 err_wrt_innerP[i] += lowerLayer.getInnerPotentialGradient()[k] * lowerLayer.getParameters().getWeights()[k][i];
            }
            err_wrt_innerP[i] = err_wrt_innerP[i] * activationFunction.derivative(innerPotencials[i]);
            
            for(int j=0; j<upperLayer.getNumberOfUnits(); j++) {
                e_wrt_weight[i][j] = err_wrt_innerP[i] * upperLayer.getOutput()[j]; // innerPotential of neuron "i" * output of neuron "j"
            }
        }
        biasErrors.add(err_wrt_innerP); // e_wrt_innerP = e_wrt_bias
        weightErrors.add(e_wrt_weight);
    }

    @Override
    public void forwardPass () {
        double[] input = upperLayer.getOutput ();
        for (int n = 0; n < numberOfUnits; n++) {
            innerPotencials[n] = bias[n];
            for (int i = 0; i < weights[n].length; i++) {
                innerPotencials[n] += input[i] * weights[n][i];
            }
            output[n] = activationFunction.apply (innerPotencials[n]);
        }
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
                weights[i][j] = r.nextGaussian ();
            }
        }
        for (int i = 0; i < bias.length; i++) {
            bias[i] = 0;
        }
    }
    
    @Override
    public void setLowerLayer (LayerWithInput nextLayer) {
        this.lowerLayer = nextLayer;
    }
    
    @Override
    public void setUpperLayer (LayerWithOutput previousLayer) {
        this.upperLayer = previousLayer;
        weights = new double[numberOfUnits][previousLayer.getNumberOfUnits ()];
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
}
