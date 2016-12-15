package cz.pv021.neuralnets.functions;

/**
 *
 * @author  Lukáš Daubner
 * @since   2016-10-30
 * @version 2016-12-14
 */
public interface HiddenFunction extends ActivationFunction {
    public double apply (double innerPotential);
    
    public double derivative (double innerPotential);
}
