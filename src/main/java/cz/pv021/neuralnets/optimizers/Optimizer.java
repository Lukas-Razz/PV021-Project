package cz.pv021.neuralnets.optimizers;

import cz.pv021.neuralnets.utils.LayerParameters;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Class handling parameter optimization
 * 
 * @author  Lukáš Daubner
 * @since   2016-12-07
 * @version 2016-12-10
 */
public class Optimizer {
    private static final Logger LOGGER = LoggerFactory.getLogger (Optimizer.class);
    
    private final double learningRate;
    private final double momentum;
    private final double l1;
    private final double l2;
    private final OptimizerAlgorithm optimizer;
    
    private Map<Integer, LayerParameters> momentumMap = new HashMap<>();
    
    public Optimizer(double learningRate, OptimizerAlgorithm optimizer, double momentum, double l1, double l2) {
        this.learningRate = learningRate;
        this.optimizer = optimizer;
        this.momentum = momentum;
        this.l1 = l1;
        this.l2 = l2;
    }
    
    protected double[][] changeWeights(double[][] weights, double[][] weightGradients, double[][] previousWeightGradients) {
        for(int i=0; i<weights.length; i++) {
            for(int j=0; j<weights[i].length; j++) {
                weights[i][j] = weights[i][j] - (learningRate * weightGradients[i][j]) - (momentum * previousWeightGradients[i][j]);
            }
        }
        return weights;
    }
    
    protected double[] changeBias(double[] bias, double[] biasGradients, double[] previousBiasGradients) {
        for(int i=0; i<bias.length; i++) {
            bias[i] = bias[i] - (learningRate * biasGradients[i]) - (momentum * previousBiasGradients[i]);
        }
        return bias;
    }
    
    private double l1Gradient(double weight) {
        return l1 * Math.signum(weight);
    }
    
    private double l2Gradient(double weight) {
        return l2 * (weight / 2);
    }
    
    public LayerParameters changeParameters(LayerParameters parameters, List<LayerParameters> gradients) {
        LayerParameters change = optimizer.computeChange(gradients);
        change = applyL1L2(parameters, change);
        
        LayerParameters previousChange = null;
        if(momentumMap.containsKey(parameters.getLayerId())) {
            previousChange = momentumMap.get(parameters.getLayerId());
        }
        else {
            previousChange = createZeroParametersFrom(parameters);
        }
        
        LayerParameters newParameters = new LayerParameters(
                changeWeights(parameters.getWeights(), change.getWeights(), previousChange.getWeights()),
                changeBias(parameters.getBias(), change.getBias(), previousChange.getBias()),
                parameters.getLayerId()
        );
        
        momentumMap.put(parameters.getLayerId(), change);
        return newParameters;
    }
    
    private LayerParameters applyL1L2(LayerParameters parameters, LayerParameters gradient) {
        for(int i=0; i<gradient.getWeights().length; i++) {
            for(int j=0; j<gradient.getWeights()[i].length; j++) {
                gradient.getWeights()[i][j] = gradient.getWeights()[i][j] + l1Gradient(parameters.getWeights()[i][j]) + l2Gradient(parameters.getWeights()[i][j]);
            }
        }
        return gradient;
    }
    
    private LayerParameters createZeroParametersFrom(LayerParameters parameters) {
        double[][] w = new double[parameters.getWeights().length][];
        double[] b = new double[parameters.getBias().length];
        for(int i=0; i<parameters.getWeights().length; i++) {
            w[i] = new double[parameters.getWeights()[i].length];
            for(int j=0; j<parameters.getWeights()[i].length; j++) {
                w[i][j] = 0;
            }
            b[i] = 0;
        }
        return new LayerParameters(w, b, parameters.getLayerId());
    }
}
