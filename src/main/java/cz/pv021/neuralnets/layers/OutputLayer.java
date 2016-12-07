package cz.pv021.neuralnets.layers;

import cz.pv021.neuralnets.error.Loss;

/**
 * Just an interface alias.
 * 
 * @author  Josef Plch
 * @since   2016-11-08
 * @version 2016-12-07
 */
public interface OutputLayer extends LayerWithInput {
    
    void setExpectedOutput(double expectedOutput);
    
    void setLoss(Loss loss);
}
