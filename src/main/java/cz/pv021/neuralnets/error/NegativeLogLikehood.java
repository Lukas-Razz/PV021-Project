package cz.pv021.neuralnets.error;

/**
 *
 * @author Lukáš Daubner
 * @since   2016-11-17
 * @version 2016-11-17
 */
public class NegativeLogLikehood implements Loss {

    @Override
    public double loss(double[] actual, double expected) {
        int classNo = (int)expected;
        return -Math.log(actual[classNo]);
    }
    
    @Override
    public double derivative(double innerPontencial)
    {
        return -(1/innerPontencial);
    }
}
