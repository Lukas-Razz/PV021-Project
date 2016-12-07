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
 * @version 2016-12-07
 */
public class Optimizer {
    private static final Logger LOGGER = LoggerFactory.getLogger (Optimizer.class);
    
    private final double learningRate;
    private final OptimizerAlgorithm optimizer;
    
    public Optimizer(double learningRate, OptimizerAlgorithm optimizer) {
        this.learningRate = learningRate;
        this.optimizer = optimizer;
    }
    
    protected double[][] changeWeights(double[][] weights, double[][] weightGradients) {
        for(int i=0; i<weights.length; i++) {
            for(int j=0; j<weights[i].length; j++) {
                weights[i][j] = weights[i][j] - learningRate * weightGradients[i][j];
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
    
    public LayerParameters changeParameters(LayerParameters parameters, List<LayerParameters> gradients) {
        LayerParameters change = optimizer.computeChange(gradients);
        
        return new LayerParameters(
                changeWeights(parameters.getWeights(), change.getWeights()),
                changeBias(parameters.getBias(), change.getBias())
        );
    }
}
