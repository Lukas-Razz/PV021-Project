package cz.pv021.neuralnets.initialization;

import cz.pv021.neuralnets.functions.ActivationFunction;
import cz.pv021.neuralnets.functions.InitializationStrategy;
import cz.pv021.neuralnets.utils.LayerParameters;

/**
 * Class handling parameter initialization
 * 
 * @author  Lukáš Daubner
 * @since   2016-12-14
 * @version 2016-12-14
 */
public class Initializer {

    private final Initialization initialization;
    
    public Initializer(Initialization initialization) {
        this.initialization = initialization;
    }
    
    public void initialize(LayerParameters parameters, ActivationFunction function) {
        if(function.getInitializationStrategy() == InitializationStrategy.SigmoidLike){
            initialization.initializeSIGMOIDLike(parameters);
        }
        else if(function.getInitializationStrategy() == InitializationStrategy.TanhLike) {
            initialization.initializeTANHLike(parameters);
        }
        else
            throw new RuntimeException("Unknown InitializationStrategy");
    }
}
