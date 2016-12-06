package cz.pv021.neuralnets.optimizers;

import cz.pv021.neuralnets.utils.LayerParameters;

/**
 * Interface for optimalization algorithm
 * 
 * @author  Lukáš Daubner
 * @since   2016-12-06
 * @version 2016-12-06
 */
public interface Optimizer {

    double[] changeBias(double[] bias, double[] biasGradients);

    LayerParameters changeParameters(LayerParameters parameters, LayerParameters gradients);

    double[][] changeWeights(double[][] weights, double[][] weightGradients);
    
}
