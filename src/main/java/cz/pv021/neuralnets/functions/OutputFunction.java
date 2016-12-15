package cz.pv021.neuralnets.functions;

/**
 *
 * @author  Lukáš Daubner
 * @since   2016-10-30
 * @version 2016-12-14
 */
public interface OutputFunction extends ActivationFunction {
    public double[] apply (double[] innerPotentials);
    
    public double[] derivative (double[] innerPotentials);
}
