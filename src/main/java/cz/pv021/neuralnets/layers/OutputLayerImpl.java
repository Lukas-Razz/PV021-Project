package cz.pv021.neuralnets.layers;

import cz.pv021.neuralnets.error.Cost;
import cz.pv021.neuralnets.error.Loss;
import cz.pv021.neuralnets.functions.OutputFunction;
import cz.pv021.neuralnets.utils.LayerParameters;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Implementation of output layer.
 * 
 * @author  Lukáš Daubner
 * @since   2016-10-30
 * @version 2016-11-17
 */
public class OutputLayerImpl implements OutputLayer {
    final Logger logger = LoggerFactory.getLogger(OutputLayerImpl.class);
    
    private final OutputFunction outputFunction;
    private final int numberOfUnits;
    private double[] output;
    private Layer upperLayer; // Vstupni
    private double[][] weights;
    private double[] bias;
    
    private double[] innerPotencials;
    
    private List<double[][]> weightErrors;
    private List<double[]> biasErrors;
    
    private Loss loss; // narvi tam loss

    public OutputLayerImpl (int numberOfUnits, OutputFunction outputFunction) {
        this.numberOfUnits = numberOfUnits;
        this.output = new double[numberOfUnits];
        this.bias = new double[numberOfUnits];
        this.outputFunction = outputFunction;
        
        this.innerPotencials = new double[numberOfUnits];
        
        weightErrors = new ArrayList<>();
        biasErrors = new ArrayList<>();
        
    }
    
    @Override
    public void backwardPass () {
        double[] e_wrt_innerP = new double[numberOfUnits];
        double[][] e_wrt_weight = new double[numberOfUnits][upperLayer.getNumberOfUnits()];
        
        for(int i=0; i<numberOfUnits; i++) {
            e_wrt_innerP[i] = loss.derivative(innerPotencials[i]);
            
            for(int j=0; j<upperLayer.getNumberOfUnits(); j++) {
                e_wrt_weight[i][j] = e_wrt_innerP[i] * upperLayer.getOutput()[j]; // innerPotential of neuron "i" * output of neuron "j"
            }
        }
        biasErrors.add(e_wrt_innerP); // e_wrt_innerP = e_wrt_bias
        weightErrors.add(e_wrt_weight);
    }
    
    @Override
    //Inner potencial is remembered for backward pass
    public void forwardPass () {
        double[] input = upperLayer.getOutput ();
        
        for (int n = 0; n < numberOfUnits; n++) {
            innerPotencials[n] = bias[n];
            for (int i = 0; i < weights[n].length; i++) {
                innerPotencials[n] += input[i] * weights[n][i];
            }
        }
        
        this.output = outputFunction.apply (innerPotencials);
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
    public void setUpperLayer (LayerWithOutput layer) {
        this.upperLayer = layer;
        this.weights = new double[numberOfUnits][layer.getNumberOfUnits ()];
    }

    @Override
    public LayerParameters getParameters() {
        return new LayerParameters(weights, bias);
    }
}
