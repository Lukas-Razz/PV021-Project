package cz.pv021.neuralnets.optimizers;

import cz.pv021.neuralnets.utils.LayerParameters;
import java.util.List;

/**
 * Interface for optimization algorithm
 * 
 * @author  Lukáš Daubner
 * @since   2016-12-06
 * @version 2016-12-07
 */
public interface OptimizerAlgorithm {
    
    LayerParameters computeChange(List<LayerParameters> gradients);
    
}
