package cz.pv021.neuralnets.optimizers;

import cz.pv021.neuralnets.utils.LayerParameters;
import java.util.List;
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
    private final double l1;
    private final double l2;
    private final OptimizerAlgorithm optimizer;
    
    public Optimizer(double learningRate, OptimizerAlgorithm optimizer, double l1, double l2) {
        this.learningRate = learningRate;
        this.optimizer = optimizer;
        this.l1 = l1;
        this.l2 = l2;
    }
    
    protected double[][] changeWeights(double[][] weights, double[][] weightGradients) {
        for(int i=0; i<weights.length; i++) {
            for(int j=0; j<weights[i].length; j++) {
                weights[i][j] = weights[i][j] - learningRate * (weightGradients[i][j] + l1Gradient(weights[i][j]) + l2Gradient(weights[i][j]));
            }
        }
        return weights;
    }
    
    protected double[] changeBias(double[] bias, double[] biasGradients) {
        for(int i=0; i<bias.length; i++) {
            bias[i] = bias[i] - learningRate * biasGradients[i];
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
        
        LayerParameters newParameters = new LayerParameters(
                changeWeights(parameters.getWeights(), change.getWeights()),
                changeBias(parameters.getBias(), change.getBias())
        );
        return newParameters;
    }
}
