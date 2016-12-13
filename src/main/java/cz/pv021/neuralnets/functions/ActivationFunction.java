package cz.pv021.neuralnets.functions;

/**
 *
 * @author  Lukáš Daubner
 * @since   2016-10-30
 * @version 2016-12-13
 */
public interface ActivationFunction {
    public double apply (double innerPotential);
    
    public double derivative (double innerPotential);
}
