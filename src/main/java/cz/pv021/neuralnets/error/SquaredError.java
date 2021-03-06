package cz.pv021.neuralnets.error;

/*
 * Implementation of Squared error loss function
 * 
 * @author  Lukáš Daubner
 * @since   2016-12-14
 * @version 2016-12-14
 */
public class SquaredError implements Loss {

    @Override
    public double loss(double actual, double expected) {
        return Math.pow((expected - actual), 2);
    }

    @Override
    public double derivative(double actual, double expected) {
        return -2*(expected - actual);
    }
    
}
