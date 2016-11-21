package cz.pv021.neuralnets.error;

/**
 * Interface of Loss (error) function
 * 
 * @author Lukáš Daubner
 * @since   2016-11-17
 * @version 2016-11-17
 */
public interface Loss {
    double loss(double[] actual, double expected);
}