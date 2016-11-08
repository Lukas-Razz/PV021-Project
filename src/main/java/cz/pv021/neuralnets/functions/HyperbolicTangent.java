package cz.pv021.neuralnets.functions;

/**
 *
 * @author  Lukáš Daubner
 * @since   2016-10-30
 * @version 2016-11-07
 */
public class HyperbolicTangent implements ActivationFunction {
    @Override
    public double apply (double innerPotencial) {
        return Math.tanh (innerPotencial);
    }

    @Override
    public double derivative (double innerPotencial) {
        // 1 - tanh^2 x = 1 / cos^2 x
        return (1 - Math.pow (Math.tanh (innerPotencial), 2));
    }
}
