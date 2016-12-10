package cz.pv021.neuralnets.error;

/**
 * Implementation of Squared error loss loss function
 * 
 * @author  Lukáš Daubner
 * @since   2016-12-06
 * @version 2016-12-06
 */
public class SquaredError implements Loss {

    @Override
    public double loss(double actual, double expected) {
        return Math.pow((expected - actual), 2)/2;
    }

    @Override
    public double derivative(double actual, double expected) {
        return -(expected - actual);
    }
    
}
