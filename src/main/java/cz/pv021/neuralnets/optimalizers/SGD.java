package cz.pv021.neuralnets.optimalizers;

import cz.pv021.neuralnets.utils.LayerParameters;

/**
 * Implementation of Stochastic Gradient Descend.
 * 
 * @author  Lukáš Daubner
 * @since   2016-11-27
 * @version 2016-11-27
 */
public class SGD {
    private final double learningRate;
    
    public SGD(double learningRate) {
        this.learningRate = learningRate;
    }
    
    public double[][] changeWeights(double[][] weights, double[][] weightGradients) {
        for(int i=0; i<weights.length; i++) {
            for(int j=0; j<weights[i].length; j++) {
                weights[i][j] = weights[i][j] - learningRate * weightGradients[i][j];
            }
        }
        return weights;
    }
    
    public double[] changeBias(double[] bias, double[] biasGradients) {
        for(int i=0; i<bias.length; i++) {
            bias[i] = bias[i] - learningRate * biasGradients[i];
        }
        return bias;
    }
    
    public LayerParameters changeParameters(LayerParameters parameters, LayerParameters gradients) {
        return new LayerParameters(
            changeWeights(parameters.getWeights(), gradients.getWeights()), 
            changeBias(parameters.getBias(), gradients.getBias())
        );
    }
}
