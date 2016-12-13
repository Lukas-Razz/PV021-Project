package cz.pv021.neuralnets.functions;

/**
 *
 * @author  Lukáš Daubner
 * @since   2016-10-30
 * @version 2016-12-13
 */
public interface OutputFunction {
    public double[] apply (double[] innerPotentials);
    
    public double[] derivative (double[] innerPotentials);
}
