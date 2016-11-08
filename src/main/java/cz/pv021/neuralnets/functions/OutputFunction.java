package cz.pv021.neuralnets.functions;

/**
 *
 * @author  Lukáš Daubner
 * @since   2016-10-30
 * @version 2016-11-07
 */
public interface OutputFunction {
    public double[] apply (double[] innerPotencials);
    
    public double[] derivative (double[] innerPotencials);
}
