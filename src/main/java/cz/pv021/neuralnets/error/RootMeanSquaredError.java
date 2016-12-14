package cz.pv021.neuralnets.error;

import static java.lang.Math.*;

/**
 * Implementation of Root mean squared error loss function
 * 
 * @author  Lukáš Daubner
 * @since   2016-12-06
 * @version 2016-12-14
 */
public class RootMeanSquaredError implements Loss {

    @Override
    public double loss(double actual, double expected) {
        return sqrt(pow((expected - actual), 2)/2);
    }

    @Override
    public double derivative(double actual, double expected) {
        return (-(expected - actual)) / 2 * sqrt(pow((expected - actual), 2)/2);
    }
    
}
