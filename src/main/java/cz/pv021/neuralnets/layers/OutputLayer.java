package cz.pv021.neuralnets.layers;

import cz.pv021.neuralnets.error.Loss;
import cz.pv021.neuralnets.functions.OutputFunction;

/**
 * Output layer interface.
 * 
 * @author  Lukáš Daubner, Josef Plch
 * @since   2016-11-08
 * @version 2016-12-13
 */

public interface OutputLayer extends LayerWithInput {
    /**
     * Default implementation.
     * 
     * @return Index of the most probable class.
     */
    public default int getOutputClassIndex () {
        double[] output = this.getOutput ();
        double maxClassWeight = Double.MIN_VALUE;
        int bestClassIndex = 0;
        
        for (int i = 0; i < output.length; i++) {
            double classWeight = output[i];
            if (classWeight > maxClassWeight) {
                maxClassWeight = classWeight;
                bestClassIndex = i;
            }
        }
        
        return bestClassIndex;
    }
    
    public void setExpectedOutput (double expectedOutput);
    
    public void setLoss (Loss loss);
    
    public OutputFunction getActivationFunction();
}
