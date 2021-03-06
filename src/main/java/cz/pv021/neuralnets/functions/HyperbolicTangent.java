package cz.pv021.neuralnets.functions;

/**
 *
 * @author  Lukáš Daubner
 * @since   2016-10-30
 * @version 2016-12-14
 */
public class HyperbolicTangent implements HiddenFunction {
    
    private final InitializationStrategy strategy = InitializationStrategy.TanhLike;
    
    @Override
    public double apply (double innerPotential) {
        return Math.tanh (innerPotential);
    }

    @Override
    public double derivative (double innerPotential) {
        // 1 - tanh^2 x = 1 / cos^2 x
        return (1 - Math.pow (Math.tanh (innerPotential), 2));
    }

    @Override
    public InitializationStrategy getInitializationStrategy() {
        return strategy;
    }
}
