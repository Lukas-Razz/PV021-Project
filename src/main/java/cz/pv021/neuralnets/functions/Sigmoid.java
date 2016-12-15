package cz.pv021.neuralnets.functions;

import static java.lang.Math.*;

/**
 *
 * @author  Lukáš Daubner
 * @since   2016-12-14
 * @version 2016-12-14
 */
public class Sigmoid implements HiddenFunction {
    
    private final InitializationStrategy strategy = InitializationStrategy.SigmoidLike;
    
    @Override
    public double apply (double innerPotential) {
        return 1/1+pow(E, -innerPotential);
    }

    @Override
    public double derivative (double innerPotential) {
        return apply(innerPotential)*(1-apply(innerPotential));
    }

    @Override
    public InitializationStrategy getInitializationStrategy() {
        return strategy;
    }
}
